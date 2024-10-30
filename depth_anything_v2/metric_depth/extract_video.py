import h5py
import os

import numpy as np
from PIL import Image


# with h5py.File(os.path.join(os.getenv("SCRATCH"), "output", "depth_A_1.h5"), "r") as f:
with h5py.File(
    os.path.join(
        "/cluster/scratch/sansara/output", "depth_scene.h5"
    ),
    "r",
) as f:
    video = f["depth"][:]

imgs = video  # [:50]
imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)) * 255

# imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
imgs = [Image.fromarray(img) for img in imgs]
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save(
    os.path.join(os.getenv("SCRATCH"), "depth.gif"),
    save_all=True,
    append_images=imgs[1:],
    duration=200,
    loop=0,
)


# # with h5py.File(os.path.join(os.getenv("SCRATCH"), "data", "A_1.h5"), "r") as f:
with h5py.File(
    os.path.join(
        "/cluster/scratch/sansara/output",
        "scene.h5",
    ),
    "r",
) as f:
    video = f["video"][:]

imgs = video  # [:50]
# imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)) * 255

# imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
imgs = [Image.fromarray(img) for img in imgs]
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save(
    os.path.join(os.getenv("SCRATCH"), "video.gif"),
    save_all=True,
    append_images=imgs[1:],
    duration=200,
    loop=0,
)
