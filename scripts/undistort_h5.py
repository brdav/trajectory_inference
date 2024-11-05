import os
import argparse
import queue
from pathlib import Path
import h5py
import cv2
import multiprocessing as mp


parser = argparse.ArgumentParser(prog="Undistort_h5")
# paths
parser.add_argument("--base-dir", type=str)
parser.add_argument("--replace-from", type=str)
parser.add_argument("--replace-to", type=str)
parser.add_argument("--log-dir", type=str, default=".")
# tuning parameters
parser.add_argument("--processes", type=int, default=64)
parser.add_argument("--chunk-size", type=int, default=48)
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
parser.add_argument("--h5-chunk-size", type=int, default=24)


def process_files(rank, args, file_queue):

    while not file_queue.empty():
        try:
            file_path = file_queue.get_nowait()
        except queue.Empty:
            break

        try:  # catch all errors

            os.makedirs(
                os.path.dirname(file_path).replace(args.replace_from, args.replace_to),
                exist_ok=True,
            )

            with h5py.File(file_path, "r") as read_f:
                with h5py.File(
                    os.path.join(
                        os.path.dirname(file_path),
                        f"camera_{os.path.basename(file_path)}",
                    ),
                    "r",
                ) as read_calib_f:
                    K = read_calib_f["camera"][:]
                    dist = read_calib_f["distortion"][:]

                with h5py.File(
                    os.path.join(
                        os.path.dirname(file_path).replace(
                            args.replace_from, args.replace_to
                        ),
                        os.path.basename(file_path),
                    ),
                    "w",
                ) as write_f:
                    new_K, _ = cv2.getOptimalNewCameraMatrix(
                        K,
                        dist,
                        (args.image_size[1], args.image_size[0]),
                        0,
                        (args.image_size[1], args.image_size[0]),
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K,
                        dist,
                        None,
                        new_K,
                        (args.image_size[1], args.image_size[0]),
                        5,
                    )

                    with h5py.File(
                        os.path.join(
                            os.path.dirname(file_path).replace(
                                args.replace_from, args.replace_to
                            ),
                            f"camera_{os.path.basename(file_path)}",
                        ),
                        "w",
                    ) as write_calib_f:
                        write_calib_f.create_dataset(
                            "camera",
                            data=new_K,
                            dtype="float32",
                        )

                    num_written = read_f["num_written"][0]

                    video_ds = write_f.create_dataset(
                        "video",
                        (num_written, args.image_height, args.image_width, 3),
                        chunks=(
                            min(args.h5_chunk_size, num_written),
                            args.image_height,
                            args.image_width,
                            3,
                        ),
                        dtype="uint8",
                    )
                    write_f.create_dataset(
                        "num_written", data=[num_written], dtype="int32"
                    )

                    read_idx = 0
                    while read_idx < num_written:
                        start_idx = read_idx
                        stop_idx = min(read_idx + args.chunk_size, num_written)
                        data = read_f["video"][start_idx:stop_idx]

                        # undistort the image
                        for i, d in enumerate(data):
                            undistort_d = cv2.remap(d, mapx, mapy, cv2.INTER_LINEAR)
                            video_ds[start_idx + i] = undistort_d

                        read_idx = stop_idx

                    assert read_idx == num_written

            print(f"Finished processing {file_path}")

        except Exception as e:
            print(e)
            print(f"Error processing {file_path}. Moving on.")
            with open(os.path.join(args.log_dir, "failed_undistort_h5.txt"), "a") as f:
                f.write(f"{file_path} REASON: {e}\n")
            try:
                os.remove(
                    os.path.join(
                        os.path.dirname(file_path).replace(
                            args.replace_from, args.replace_to
                        ),
                        os.path.basename(file_path),
                    )
                )
            except OSError:
                pass
            try:
                os.remove(
                    os.path.join(
                        os.path.dirname(file_path).replace(
                            args.replace_from, args.replace_to
                        ),
                        f"camera_{os.path.basename(file_path)}",
                    )
                )
            except OSError:
                pass


if __name__ == "__main__":

    args = parser.parse_args()

    file_paths = sorted(Path(args.base_dir).rglob("*.h5"))

    file_queue = mp.Queue()
    for p in file_paths:
        file_queue.put(p)

    processes = []
    for rank in range(args.processes):
        p = mp.Process(
            target=process_files,
            args=(
                rank,
                args,
                file_queue,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
