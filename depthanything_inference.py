import os
import argparse
import queue
import h5py
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity
from torchvision.transforms import Compose

from depth_anything_v2.metric_depth.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)
from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


parser = argparse.ArgumentParser(prog="DepthAnything_inference")
# paths
parser.add_argument("--save-dir", type=str, default="./output")
parser.add_argument("--file-list", type=str, default="./h5_file_list.txt")
parser.add_argument("--weights-dir", type=str, default="./weights")
# tuning parameters
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--num-proc-per-gpu", type=int, default=1)
parser.add_argument("--buffer-size", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=16)
# constants
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
parser.add_argument("--h5-chunk-size", type=int, default=24)
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="unidepth_exp")


class H5Dataset(Dataset):

    def __init__(self, input_size=518):
        self.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # why fill dataset at getitem rather than init?
        # each worker (which are forked after the init) need to have their own file handle
        if self.dataset is None:
            self.file = h5py.File(self.file_path, "r")
            self.dataset = self.file.get("video")
        img = self.dataset[idx] / 255.0
        return self.transform({"image": img})["image"]

    def load_new_file(self, file_path):
        self.dataset = None
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.len = f.get("num_written")[0]


def process_files(rank, p_rank, args, file_queue, model):

    # just in case DepthAnything internals use default
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
    model = model.to(f"cuda:{rank}").eval()

    # each process assigns num_workers workers for data loading
    # if num_workers=0, main process handles loading
    dataset = H5Dataset()
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
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
            os.path.join(args.save_dir, f"depth_{os.path.basename(file_path)}")
        ):
            try:
                with h5py.File(
                    os.path.join(args.save_dir, f"depth_{os.path.basename(file_path)}"),
                    "r",
                ) as depth_file:
                    num_written = depth_file["num_written"][0]
                    depth = depth_file["depth"][:]
                assert num_written == len(depth)
                print(
                    f"Depth H5 file for {os.path.basename(file_path)} already processed, skipping!"
                )
                continue
            except Exception as e:
                print(
                    f"Depth H5 file for {os.path.basename(file_path)} seems to be corrupt. Will overwrite."
                )
                os.remove(
                    os.path.join(args.save_dir, f"depth_{os.path.basename(file_path)}")
                )

        # open target h5
        with h5py.File(
            os.path.join(args.save_dir, f"depth_{os.path.basename(file_path)}"), "w"
        ) as depth_file:

            depth_ds = depth_file.create_dataset(
                "depth",
                (dataset.len, args.image_height, args.image_width),
                chunks=(
                    min(args.h5_chunk_size, dataset.len),
                    args.image_height,
                    args.image_width,
                ),
                dtype="float32",
            )
            depth_file.create_dataset("num_written", data=[dataset.len], dtype="int32")

            write_idx = 0
            depth_pred = []
            for data in data_loader:
                predictions = model.infer_batch(
                    data.to(f"cuda:{rank}"), [args.image_height, args.image_width]
                )
                depth_pred.append(predictions)
                if len(depth_pred) * args.batch_size >= args.buffer_size:
                    # dump buffer to file
                    depth_pred_np = torch.cat(depth_pred, dim=0).cpu().numpy()
                    depth_ds[write_idx : write_idx + len(depth_pred_np)] = depth_pred_np
                    write_idx += len(depth_pred_np)
                    depth_pred = []

                if (not args.no_profiler) and (rank == 0):
                    prof.step()

            # dump the rest
            depth_pred_np = torch.cat(depth_pred, dim=0).cpu().numpy()
            depth_ds[write_idx:] = depth_pred_np

        print(f"Finished processing {file_path} on gpu {rank} process {p_rank}")

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
    print(f"buffer-size: {args.buffer_size}")
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

    # model config
    encoder = "vitl"  # or 'vits', 'vitb'
    dataset = "vkitti"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80  # 20 for indoor model, 80 for outdoor model
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.weights_dir, f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
            ),
            map_location="cpu",
        )
    )

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
