import time
from pathlib import Path
import hashlib

import numpy as np

import rosu_pp_py as rosu
from .beatmaps.beatmap import Beatmap

from .beatmaps.encode import encode_beatmap
from .dac import encode, load_dac, get_frame_times

NUM_LABELS = 1

perf = rosu.Performance()
dac = load_dac("no_dec_16kbps_44khz.safetensors")


def prepare_map(data_dir: Path, map_file: Path):
    start = time.time()
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    if bm.mode != 0:
        # not osu!std, skip
        # print(f"{map_file}: not an osu!std map")
        return

    af_dir = "_".join(
        [bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)]
    )
    hash = hashlib.md5(
        map_file.parent.name.encode("utf8"), usedforsecurity=False
    ).hexdigest()
    map_dir = data_dir / hash / af_dir

    spec_path = map_dir / "spec.pt"
    map_path = map_dir / f"{map_file.stem}.map.pt"

    if map_path.exists():
        return

    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    # difficulty calculation
    diff_attrs = perf.calculate(rosu.Beatmap(path=str(map_file))).difficulty
    diff_labels = np.array([diff_attrs.stars])
    assert len(diff_labels) == NUM_LABELS

    if spec_path.exists():
        for i in range(5):
            try:
                spec = np.load(spec_path)
                break
            except (ValueError, EOFError):
                # can be raised if file was created but writing hasn't completed
                # just wait a little for the writing to finish
                time.sleep(0.01 * 2**i)
        else:
            # retried 5 times without success, just skip
            print(f"{bm.audio_filename}: unable to load spectrogram from {spec_path}")
            return
    else:
        # load audio file
        try:
            spec = (
                encode(dac, bm.audio_filename, chunk_size=4)[0, :, :].detach().numpy()
            )
        except Exception as e:
            print(f"{bm.audio_filename}: {e}")
            return

        # save spectrogram
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)

    frame_times = get_frame_times(spec, dac)

    # compute map signal
    try:
        x = encode_beatmap(bm, frame_times)
    except Exception as e:
        print(e)
        raise RuntimeError("failed to encode beatmap")

    with open(map_path, "wb") as f:
        for obj in [x, diff_labels]:
            np.save(f, obj, allow_pickle=False)
    print(f"Processed {map_file.stem} in {time.time() - start:.2f}s.")
