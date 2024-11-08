import os
import h5py
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from p_tqdm import p_map
import os
import numpy as np

BASE_DIR = Path("/store/swissai/a03/datasets")

H5_DIRS = [
    "CCD/h5_files",
    "D2City",
    "DAD",
    "DoTA",
    "Drive360",
    # "DrivingDojo_h5",  # undistort first --> README
    "HondaHAD",
    # "ONCE",  # undistort first --> README
    "OpenDV-YouTube/h5",
    "YouTubeCrash",
    "bdd100k",
    "cityscapes_h5",
    # "dad_streetaccident",
    "kitti_h5",
    "nuplan_h5",
    # "nuscenes_h5",
    # "HONDAHDD",  # not ready yet
]

def check_status_of_depth(file):
    try:
        with h5py.File(file, "r") as depth_file:
            num_written = depth_file["num_written"][0]
            tmp = depth_file["depth"][num_written - 1]
        assert not np.array_equal(tmp, np.zeros_like(tmp))
        with open("good_files.txt", "a") as f:
            f.write(f"{file}\n")
        return True
    except Exception as e:
        print(e)
        print(f"Error opening {file}. Moving on.")
        with open("bad_files.txt", "a") as f:
            f.write(f"{file}\n")
        return False

def collect_frames():
    for dataset in tqdm(H5_DIRS):
        dirpath = BASE_DIR / (dataset + "_proc")
        paths = list(dirpath.rglob("*.h5"))
        # Filter out all paths whose filenames start with camera_, depth_ or trajectory_
        paths = [p for p in paths if any([p.name.startswith(x) for x in ["depth_"]])]
        statuses = p_map(check_status_of_depth, paths)
        # print ratio of completion 
        print(f"Dataset: {dataset}, {sum(statuses)}/{len(statuses)*100:.2f}%")

def main():
    collect_frames()

if __name__ == "__main__":
    main()
