import os
import numpy as np
import cv2
import torch
import torch.nn as nn
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


class TrajectoryExtractor:

    def __init__(self,
                 image_height=576,
                 image_width=1024,
                 encoder="vitl",
                 weights_dir="."):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_height = image_height
        self.image_width = image_width
        self.weights_dir = weights_dir

        # load calib model
        self.calib_model = GeoCalib()
        self.calib_model = self.calib_model.to(self.device)

        # load depth model
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
        self.depth_model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
        self.depth_model.load_state_dict(
            torch.load(
                os.path.join(
                    weights_dir,
                    f"depth_anything_v2_metric_{dataset}_{encoder}.pth",
                ),
                map_location="cpu",
            )
        )
        self.depth_model = self.depth_model.to(self.device).eval()
        self.depth_transform = Compose(
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

        # resize and crop as in DroidSLAM repo
        self.resize_height = int(
            image_height
            * np.sqrt((384 * 512) / (image_height * image_width))
        )
        self.resize_width = int(
            image_width * np.sqrt((384 * 512) / (image_height * image_width))
        )
        self.crop_height = self.resize_height - self.resize_height % 8
        self.crop_width = self.resize_width - self.resize_width % 8

    def __call__(self, video):
        calib_matrix = self.do_calib(video)
        depth_video = self.do_depth(video)
        trajectory = self.do_slam(video, depth_video, calib_matrix)
        # return 2d trajectory
        return np.array([trajectory[:, 0, 3], trajectory[:, 2, 3]]).T  # N x 2

    @torch.no_grad()
    def do_calib(self, video):
        video = (video.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
        video = torch.from_numpy(video).to(self.device)
        result = self.calib_model.calibrate(
            video,
            camera_model="pinhole",
            shared_intrinsics=True,
        )
        intrinsics_np = result["camera"].K.mean(dim=0).cpu().numpy()
        return intrinsics_np

    @torch.no_grad()
    def do_depth(self, video):
        video = video / 255.0
        video = np.array([self.depth_transform({"image": d})["image"] for d in video])
        video = torch.from_numpy(video).to(self.device)
        predictions = self.depth_model.forward(video)
        predictions = nn.functional.interpolate(
            predictions[:, None],
            [self.image_height, self.image_width],
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        predictions_np = predictions.cpu().numpy()
        return predictions_np

    @torch.no_grad()
    def do_slam(self, video, depth_video, calib_matrix):
        fx = calib_matrix[0, 0] * self.resize_width / self.image_width
        fy = calib_matrix[1, 1] * self.resize_height / self.image_height
        cx = calib_matrix[0, 2] * self.resize_width / self.image_width
        cy = calib_matrix[1, 2] * self.resize_height / self.image_height
        intrinsics = np.array([fx, fy, cx, cy])

        filter_thresh = 2.4
        frontend_thresh = 16.0
        backend_thresh = 22.0
        keyframe_thresh = 4.0
        while True:
            try:
                droid = Droid(
                    weights=f"{self.weights_dir}/droid.pth",
                    image_size=[self.crop_height, self.crop_width],
                    upsample=True,
                    buffer=512,
                    device=self.device,
                    filter_thresh=filter_thresh,
                    frontend_thresh=frontend_thresh,
                    backend_thresh=backend_thresh,
                    keyframe_thresh=keyframe_thresh,
                )

                for idx, image, depth, intr in self.image_stream(video, depth_video, intrinsics, [self.resize_height, self.resize_width], [self.crop_height, self.crop_width]):
                    droid.track(idx, image, depth, intr)

                # do global bundle adjustment
                traj_est = droid.terminate(self.image_stream(video, depth_video, intrinsics, [self.resize_height, self.resize_width], [self.crop_height, self.crop_width]))
                break
            except Exception as e:
                if filter_thresh > 0.1:
                    filter_thresh /= 2.0
                    frontend_thresh /= 2.0
                    backend_thresh /= 2.0
                    keyframe_thresh /= 2.0
                else:
                    raise e

        traj_est = self.get_pose_matrix(traj_est)
        return traj_est

    def get_pose_matrix(self, traj):
        Ts = []
        for i in range(len(traj)):
            pose = traj[i]
            t, q = pose[1:4], pose[4:]
            R = self.quaternion_to_matrix(q)
            T = np.eye(4)
            # Twc = [R | t]
            T[:3, :3] = R
            T[:3, 3] = t
            Ts.append(T)
        return np.stack(Ts, axis=0)

    @staticmethod
    def quaternion_to_matrix(q):
        Q = SO3.InitFromVec(torch.Tensor(q))
        R = Q.matrix().detach().cpu().numpy().astype(np.float32)
        return R[:3, :3]

    @staticmethod
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


if __name__ == "__main__":

    import h5py
    import matplotlib.pyplot as plt

    file_paths = [
        "$SCRATCH/generated/chunk_0.h5",
        "$SCRATCH/generated/chunk_1.h5",
        "$SCRATCH/generated/chunk_2.h5",
        "$SCRATCH/generated/chunk_3.h5",
        "$SCRATCH/generated/chunk_4.h5",
    ]

    extractor = TrajectoryExtractor(encoder="vitl", weights_dir="/capstor/scratch/cscs/pmartell/trajectory_inference/weights")

    for file_path in file_paths:
        print(f"doing {file_path}...")
        with h5py.File(file_path, "r") as f:
            video = f["video"][:]

        # feed in video as N x H x W x 3
        trajectory_2d = extractor(video)  # returns N x 2 trajectory

        plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1])
        plt.gca().set_aspect('equal')
        plt.savefig(f"trajectory_{os.path.basename(file_path)}.png")
        plt.close()
