import os
import argparse
import warnings
import queue
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.profiler import profile, ProfilerActivity

from geocalib import GeoCalib


parser = argparse.ArgumentParser(prog="GeoCalib_inference")
# paths
parser.add_argument("--save-dir", type=str, default="./output")
parser.add_argument("--file-list", type=str, default="./h5_file_list.txt")
# tuning parameters
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--num-proc-per-gpu", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--camera-model", type=str, default="pinhole")
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="geocalib_exp")


def process_files(rank, p_rank, args, file_queue, model):

    # just in case GeoCalib internals use default
    torch.cuda.set_device(f"cuda:{rank}")

    # run profiler on rank 0 GPU only
    if (not args.no_profiler) and (rank == 0):
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.log_dir, f"{args.exp_name}_rank_{rank}_{p_rank}")
            ),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        prof.start()

    # push model to assigned GPU
    model = model.to(f"cuda:{rank}")

    while not file_queue.empty():
        try:
            file_path = file_queue.get_nowait()
        except queue.Empty:
            break

        # check if file is already processed
        if os.path.exists(
            os.path.join(args.save_dir, f"camera_{os.path.basename(file_path)}")
        ):
            try:
                with h5py.File(
                    os.path.join(
                        args.save_dir, f"camera_{os.path.basename(file_path)}"
                    ),
                    "r",
                ) as calib_file:
                    K = calib_file["camera"][:]
                assert K.shape == (3, 3)
                print(
                    f"Calib H5 file for {os.path.basename(file_path)} already processed, skipping!"
                )
                continue
            except Exception as e:
                print(
                    f"Calib H5 file for {os.path.basename(file_path)} seems to be corrupt. Will overwrite."
                )
                os.remove(
                    os.path.join(args.save_dir, f"camera_{os.path.basename(file_path)}")
                )

        # open target h5
        with h5py.File(
            os.path.join(args.save_dir, f"camera_{os.path.basename(file_path)}"), "w"
        ) as calib_file:

            with h5py.File(file_path, "r") as video_file:
                length = video_file.get("num_written")[0]
                grid_points = torch.unique(torch.round(
                    torch.linspace(
                            0, length - 1, steps=args.batch_size + 2
                        )[1:-1]
                    ).to(int))
                data = video_file.get("video")[grid_points]
                
            data = (data.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
            data = torch.from_numpy(data).to(f"cuda:{rank}")

            if args.camera_model == "pinhole":
                result = model.calibrate(
                    data, camera_model="pinhole", shared_intrinsics=True
                )
                # no distortion parameters because we use pinhole model
                K = result["camera"].K
                # roll and pitch angles of the gravity vector
                rp = result["gravity"].rp
            else:
                # batched inference is unfortunately not possible
                results = [model.calibrate(d, camera_model=args.camera_model) for d in data]
                K = torch.stack([r["camera"].K for r in results], dim=0)
                rp = torch.stack([r["gravity"].rp for r in results], dim=0)
                k1 = torch.stack([r["camera"].k1 for r in results], dim=0)

            # main field
            calib_file.create_dataset(
                "camera",
                data=K.mean(dim=0).cpu().numpy(),
                dtype="float32",
            )

            # if we don't use pinhole model, save distortion
            if args.camera_model != "pinhole":
                calib_file.create_dataset(
                    "distortion_k1",
                    data=k1.mean(dim=0).cpu().numpy(),
                    dtype="float32",
                )

            # save gravity
            calib_file.create_dataset(
                "gravity",
                data=rp.mean(dim=0).cpu().numpy(),
                dtype="float32",
            )

        print(f"Finished processing {file_path} on gpu {rank} process {p_rank}")

        if (not args.no_profiler) and (rank == 0):
            prof.step()

    if (not args.no_profiler) and (rank == 0):
        prof.stop()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()

    print("Starting job with the following parameters:")
    print(f"exp-name: {args.exp_name}")
    print(f"num-gpus: {args.num_gpus}")
    print(f"num-proc-per-gpu: {args.num_proc_per_gpu}")
    print(f"batch-size: {args.batch_size}")

    if args.file_list.endswith(".h5"):
        file_paths = [args.file_list]
    else:
        with open(args.file_list, "r") as f:
            file_paths = f.read().splitlines()
    file_queue = mp.Queue()

    for file_path in file_paths:
        file_queue.put(file_path)

    if args.camera_model == "pinhole":
        model = GeoCalib()
    else:
        model = GeoCalib(weights="distorted")

    processes = []
    for rank in range(args.num_gpus):
        for p_rank in range(args.num_proc_per_gpu):
            p = mp.Process(
                target=process_files,
                args=(
                    rank,
                    p_rank,
                    args,
                    file_queue,
                    model,
                ),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
