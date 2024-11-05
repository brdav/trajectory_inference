import os
import h5py
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from p_tqdm import p_map


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
    "nuscenes_h5",
    # "HONDAHDD",  # not ready yet
]

NODES = 256

def collect_path(path):
    try:
        with h5py.File(path, "r") as f:
            num_written = f["num_written"][0].item()
            return (path, num_written)
    except Exception as e:
        print(e)
        print(f"Error opening {path}. Moving on.")
        with open("output/not_included_h5.txt", "a") as f:
            f.write(f"{path} REASON: {e}\n")
        return None, None

def collect_frames():
    # if path_to_frames.json exists, load it
    if os.path.exists("output/path_to_frames.json"):
        with open("output/path_to_frames.json", "r") as f:
            return json.load(f)

    path_to_frames = {}
    for dataset in tqdm(H5_DIRS):
        print(BASE_DIR / dataset)
        tuples = p_map(collect_path, list((BASE_DIR / dataset).rglob("*.h5")))
        for p, n in tuples:
            if n is not None:
                path_to_frames[str(p)] = n

    with open("output/path_to_frames.json", "w", encoding="utf-8") as f:
        json.dump(path_to_frames, f, ensure_ascii=False, indent=4)

    return path_to_frames


def create_txt(paths_per_node):
    for k, v in paths_per_node.items():
        with open(f"output/file_list_node_{k}.txt", "w") as f:
            for p in v:
                f.write(f"{p}\n")


def main():
    os.makedirs("output", exist_ok=True)
    path_to_frames = collect_frames()

    # Save path_to_frames to file


    total_frames = sum(list(path_to_frames.values()))
    frames_per_node = total_frames / NODES

    paths_per_node = defaultdict(list)
    node = 0
    curr_frames = 0
    for k, v in path_to_frames.items():
        paths_per_node[node].append(k)
        curr_frames += v
        if curr_frames >= frames_per_node:
            node += 1
            curr_frames = 0

    with open("output/paths_per_node.json", "w", encoding="utf-8") as f:
        json.dump(paths_per_node, f, ensure_ascii=False, indent=4)

    create_txt(paths_per_node)


if __name__ == "__main__":
    main()
