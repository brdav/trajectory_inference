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
parser.add_argument("--num-workers", type=int, default=1)
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="geocalib_exp")


class RegularSampler(Sampler):

    def __init__(self, data_source, num_grid_points):
        self.data_source = data_source
        self.num_grid_points = num_grid_points  # fixed

    def __len__(self):
        return self.num_grid_points

    def __iter__(self):
        grid_points = torch.round(
            torch.linspace(
                0, len(self.data_source) - 1, steps=self.num_grid_points + 2
            )[1:-1]
        ).to(int)
        return iter(grid_points)


class H5Dataset(Dataset):

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # why fill dataset at getitem rather than init?
        # each worker (which are forked after the init) need to have their own file handle
        if self.dataset is None:
            self.file = h5py.File(self.file_path, "r")
            self.dataset = self.file.get("video")
        return (self.dataset[idx].transpose(2, 0, 1) / 255.0).astype(np.float32)

    def load_new_file(self, file_path):
        self.dataset = None
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.len = f.get("num_written")[0]


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

    # each process assigns num_workers workers for data loading
    # if num_workers=0, main process handles loading
    dataset = H5Dataset()
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RegularSampler(dataset, args.batch_size),
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    while not file_queue.empty():
        try:
            file_path = file_queue.get_nowait()
        except queue.Empty:
            break

        # assign file to dataloader
        dataset.load_new_file(file_path)

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

            # grab a single batch, randomly sampled from video
            data = next(iter(data_loader)).to(f"cuda:{rank}")
            result = model.calibrate(
                data, camera_model="pinhole", shared_intrinsics=True
            )

            # no distortion parameters because we use pinhole model
            K = result["camera"].K
            # roll and pitch angles of the gravity vector
            rp = result["gravity"].rp

            # main field
            calib_file.create_dataset(
                "camera",
                data=K[0].cpu().numpy(),
                dtype="float32",
            )

            # secondary fields
            # specific to sampled images, so we save the mean
            calib_file.create_dataset(
                "focal_uncertainty",
                data=result["focal_uncertainty"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "vfov_uncertainty",
                data=result["vfov_uncertainty"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "gravity",
                data=rp.mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "covariance",
                data=result["covariance"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "up_field",
                data=result["up_field"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "latitude_field",
                data=result["latitude_field"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "up_confidence",
                data=result["up_confidence"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "latitude_confidence",
                data=result["latitude_confidence"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "roll_uncertainty",
                data=result["roll_uncertainty"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "pitch_uncertainty",
                data=result["pitch_uncertainty"].mean(dim=0).cpu().numpy(),
                dtype="float32",
            )
            calib_file.create_dataset(
                "gravity_uncertainty",
                data=result["gravity_uncertainty"].mean(dim=0).cpu().numpy(),
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
    print(f"num-workers: {args.num_workers}")

    if args.file_list.endswith(".h5"):
        file_paths = [args.file_list]
    else:
        with open(args.file_list, "r") as f:
            file_paths = f.read().splitlines()
    file_queue = mp.Queue()

    for file_path in file_paths:
        file_queue.put(file_path)

    model = GeoCalib()

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
