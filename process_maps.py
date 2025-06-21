import argparse
from concurrent import futures
from pathlib import Path


parser = argparse.ArgumentParser(
    "Map preprocessor",
    description="Process maps for training. Uses an encoder-only version of Descript Audio Codec 44KHz 18 codebook pretrained model, and osu-dreamer!s map processing.",
)
parser.add_argument(
    "song_dir",
    type=str,
    help="Directory containing the songs to process. Must be an osu!stable Songs directory.",
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="Directory to save the processed data to. Will be created if it does not exist.",
    default="data/",
)
parser.add_argument(
    "-n",
    "--num-workers",
    type=int,
    default=4,
    help="Number of workers to use for processing maps.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    from src.prepare_map import prepare_map

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    song_dir = Path(args.song_dir)
    if not song_dir.exists() or not song_dir.is_dir():
        raise ValueError(f"Invalid song directory: {song_dir}")

    with futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        n_queue = 0
        for map_file in song_dir.rglob("*.osu"):
            if not map_file.is_file():
                continue

            # TODO: this is really bad, because sometimes the same songs spec gets
            # processed multiple times, which is pretty~ wasteful
            executor.submit(prepare_map, data_dir, map_file)
            n_queue += 1
        print(f"Queued {n_queue} maps for processing.")
    print("Done processing maps.")
