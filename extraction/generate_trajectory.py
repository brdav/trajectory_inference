import os
import argparse
import warnings
import queue
import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
from torchvision.transforms import Compose
from lietorch import SO3

from geocalib import GeoCalib
from depth_anything_v2.metric_depth.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)
from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from droid_trajectory.droid_core.droid import Droid


parser = argparse.ArgumentParser(prog="Evaluation")
# paths
parser.add_argument("--file-list", type=str, default="./file_list.txt")
parser.add_argument("--replace-from", type=str, default="")
parser.add_argument("--replace-to", type=str, default="")
parser.add_argument("--log-dir", type=str, default="./extraction_logs")
parser.add_argument("--weights-dir", type=str, default=".")
parser.add_argument("--h5-chunk-size", type=int, default=24)
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
parser.add_argument("--encoder", type=str, default="vitl")


def quaternion_to_matrix(q):
    Q = SO3.InitFromVec(torch.Tensor(q))
    R = Q.matrix().detach().cpu().numpy().astype(np.float32)
    return R[:3, :3]


def get_pose_matrix(traj):
    Ts = []
    for i in range(len(traj)):
        pose = traj[i]
        t, q = pose[1:4], pose[4:]
        R = quaternion_to_matrix(q)
        T = np.eye(4)
        # Twc = [R | t]
        T[:3, :3] = R
        T[:3, 3] = t
        Ts.append(T)
    return np.stack(Ts, axis=0)


def image_stream(video, depth_video, intrinsics, resize_size, crop_size):
    for idx, (image, depth) in enumerate(zip(video, depth_video)):
        image = cv2.resize(image, (resize_size[1], resize_size[0]))
        image = image[: crop_size[0], : crop_size[1]]
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)

        depth = torch.as_tensor(depth)
        depth = nn.functional.interpolate(depth[None, None], resize_size).squeeze()
        depth = depth[: crop_size[0], : crop_size[1]]
        yield idx, image[None], depth, torch.from_numpy(intrinsics)


def do_calib(file_paths, args):

    print("Doing calib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = GeoCalib()
    model = model.to(device)

    for file_path in file_paths:

        try:  # catch all errors
            out_dir = os.path.dirname(file_path).replace(args.replace_from, args.replace_to)
            os.makedirs(out_dir, exist_ok=True)

            # check if file is already processed
            if os.path.exists(
                os.path.join(out_dir, f"camera_{os.path.basename(file_path)}")
            ):
                try:
                    with h5py.File(
                        os.path.join(
                            out_dir,
                            f"camera_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as calib_file:
                        K = calib_file["camera"][:]
                    assert K.shape == (3, 3)
                    print(
                        f"Calib H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Calib H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            out_dir,
                            f"camera_{os.path.basename(file_path)}",
                        )
                    )

            # load the data
            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]
                video = f["video"][:]
                assert len(video) == num_written

            # open target h5
            with h5py.File(
                os.path.join(out_dir, f"camera_{os.path.basename(file_path)}"),
                "w",
            ) as calib_file:

                video = (video.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
                video = torch.from_numpy(video).to(device)

                result = model.calibrate(
                    video,
                    camera_model="pinhole",
                    shared_intrinsics=True,
                )
                # no distortion parameters because we use pinhole model
                K = result["camera"].K
                # roll and pitch angles of the gravity vector
                rp = result["gravity"].rp

                calib_file.create_dataset(
                    "camera",
                    data=K.mean(dim=0).cpu().numpy(),
                    dtype="float32",
                )
                calib_file.create_dataset(
                    "gravity",
                    data=rp.mean(dim=0).cpu().numpy(),
                    dtype="float32",
                )

            print(f"Finished calib for {file_path}")

        except Exception as e:
            print(e)
            print(f"Error processing calib for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, "failed_calib.txt"), "a"
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")


def do_depth(file_paths, args):

    print("Doing depth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
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
    model = DepthAnythingV2(**{**model_configs[args.encoder], "max_depth": max_depth})
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.weights_dir,
                f"depth_anything_v2_metric_{dataset}_{args.encoder}.pth",
            ),
            map_location="cpu",
        )
    )
    model = model.to(device).eval()
    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
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

    for file_path in file_paths:

        try:  # catch all errors
            out_dir = os.path.dirname(file_path).replace(args.replace_from, args.replace_to)
            os.makedirs(out_dir, exist_ok=True)

            # check if file is already processed
            if os.path.exists(
                os.path.join(out_dir, f"depth_{os.path.basename(file_path)}")
            ):
                try:
                    with h5py.File(
                        os.path.join(
                            out_dir,
                            f"depth_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as depth_file:
                        num_written = depth_file["num_written"][0]
                        tmp = depth_file["depth"][num_written - 1]
                    assert not np.array_equal(tmp, np.zeros_like(tmp))
                    print(
                        f"Depth H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f">>>>>>>>>>>> Depth H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            out_dir,
                            f"depth_{os.path.basename(file_path)}",
                        )
                    )

            # load the data
            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]
                video = f["video"][:]
                assert len(video) == num_written

            # open target h5
            with h5py.File(
                os.path.join(out_dir, f"depth_{os.path.basename(file_path)}"),
                "w",
            ) as depth_file:

                video = video / 255.0
                video = np.array([transform({"image": d})["image"] for d in video])
                video = torch.from_numpy(video).to(device)

                with torch.no_grad():
                    predictions = model.forward(video)
                    predictions = nn.functional.interpolate(
                        predictions[:, None],
                        [args.image_height, args.image_width],
                        mode="bilinear",
                        align_corners=True,
                    )[:, 0]
                    predictions_np = predictions.cpu().numpy()

                depth_file.create_dataset(
                    "depth",
                    data=predictions_np,
                    chunks=(
                        min(args.h5_chunk_size, num_written),
                        args.image_height,
                        args.image_width,
                    ),
                    dtype="float32",
                )
                depth_file.create_dataset(
                    "num_written", data=[num_written], dtype="int32"
                )

            print(f"Finished depth for {file_path}")

        except Exception as e:
            print(e)
            print(f"Error processing depth for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, "failed_depth.txt"), "a"
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")


def do_slam(file_paths, args):

    print("Doing slam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # resize and crop as in DroidSLAM repo
    resize_height = int(
        args.image_height
        * np.sqrt((384 * 512) / (args.image_height * args.image_width))
    )
    resize_width = int(
        args.image_width * np.sqrt((384 * 512) / (args.image_height * args.image_width))
    )
    crop_height = resize_height - resize_height % 8
    crop_width = resize_width - resize_width % 8

    for file_path in file_paths:

        try:  # catch all errors
            out_dir = os.path.dirname(file_path).replace(args.replace_from, args.replace_to)
            os.makedirs(out_dir, exist_ok=True)

            # check if file is already processed
            if os.path.exists(
                os.path.join(out_dir, f"trajectory_{os.path.basename(file_path)}")
            ):

                try:
                    with h5py.File(
                        os.path.join(
                            out_dir,
                            f"trajectory_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as trajectory_file:
                        num_written = trajectory_file["num_written"][0]
                        tmp = trajectory_file["trajectory"][0, num_written - 1]
                    assert not np.array_equal(tmp, np.zeros_like(tmp))
                    print(
                        f"Trajectory H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Trajectory H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            out_dir,
                            f"trajectory_{os.path.basename(file_path)}",
                        )
                    )

            # load data
            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]
                video = f["video"][:]
                assert len(video) == num_written

            with h5py.File(os.path.join(out_dir, f"camera_{os.path.basename(file_path)}"), "r") as calib_file:
                K = calib_file["camera"][:]
                # rescale intrinsics
                fx = K[0, 0] * resize_width / args.image_width
                fy = K[1, 1] * resize_height / args.image_height
                cx = K[0, 2] * resize_width / args.image_width
                cy = K[1, 2] * resize_height / args.image_height
                intrinsics = np.array([fx, fy, cx, cy])

            with h5py.File(os.path.join(out_dir, f"depth_{os.path.basename(file_path)}"), "r") as depth_file:
                depth_video = depth_file["depth"][:]

            # open target h5
            with h5py.File(
                os.path.join(
                    out_dir,
                    f"trajectory_{os.path.basename(file_path)}",
                ),
                "w",
            ) as trajectory_file:

                filter_thresh = 2.4
                frontend_thresh = 16.0
                backend_thresh = 22.0
                keyframe_thresh = 4.0
                while True:
                    try:
                        droid = Droid(
                            weights=f"{args.weights_dir}/droid.pth",
                            image_size=[crop_height, crop_width],
                            upsample=True,
                            buffer=512,
                            device=device,
                            filter_thresh=filter_thresh,
                            frontend_thresh=frontend_thresh,
                            backend_thresh=backend_thresh,
                            keyframe_thresh=keyframe_thresh,
                        )

                        for idx, image, depth, intr in image_stream(video, depth_video, intrinsics, [resize_height, resize_width], [crop_height, crop_width]):
                            droid.track(idx, image, depth, intr)

                        # do global bundle adjustment
                        traj_est = droid.terminate(image_stream(video, depth_video, intrinsics, [resize_height, resize_width], [crop_height, crop_width]))
                        break
                    except Exception as e:
                        if filter_thresh > 0.1:
                            filter_thresh /= 2.0
                            frontend_thresh /= 2.0
                            backend_thresh /= 2.0
                            keyframe_thresh /= 2.0
                        else:
                            raise e

                traj_est = get_pose_matrix(traj_est)

                trajectory_file.create_dataset(
                    "trajectory",
                    data=traj_est[None],
                    dtype="float32",
                )
                trajectory_file.create_dataset(
                    "start_idx",
                    data=[0],
                    dtype="int32",
                )
                trajectory_file.create_dataset(
                    "stop_idx",
                    data=[len(traj_est)],
                    dtype="int32",
                )
                trajectory_file.create_dataset(
                    "num_written", data=[num_written], dtype="int32"
                )

            print(f"Finished trajectory for {file_path}")

        except Exception as e:
            print(e)
            print(f"Error processing trajectory for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, "failed_trajectory.txt"),
                "a",
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")


if __name__ == "__main__":

    args = parser.parse_args()

    print("Starting trajectory extraction")

    if args.file_list.endswith(".h5"):
        file_paths = [args.file_list]
    else:
        with open(args.file_list, "r") as f:
            file_paths = f.read().splitlines()

    os.makedirs(args.log_dir, exist_ok=True)

    do_calib(file_paths, args)
    do_depth(file_paths, args)
    do_slam(file_paths, args)
