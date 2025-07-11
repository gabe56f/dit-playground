from jaxtyping import Float
from typing import Union

import numpy as np
from numpy import ndarray

from .beatmap import Beatmap
from .hit_objects import Slider, Spinner
from .timing import FrameTimes

from scipy.signal import find_peaks

from enum import IntEnum

Real = Union[int, float]

# == events ==


def events(ts: list[Real], frame_times: FrameTimes) -> Float[ndarray, "L"]:
    """returns time (in log ms) since last event, scaled+shifted to [0,1]"""
    MIN_TIME = 4  # intervals shorter than 2^MIN_TIME milliseconds get aliased to 0
    MAX_TIME = 11  # intervals longer  than 2^MAX_TIME milliseconds get aliased to 1

    time_since_last_event = np.full_like(frame_times, 2**MAX_TIME)
    for t in ts:
        time_since_event = frame_times - t
        region = (time_since_event >= 0) & (time_since_event <= 2**MAX_TIME)
        time_since_last_event[region] = time_since_event[region]

    log_time = np.log2(time_since_last_event + 2**MIN_TIME).clip(MIN_TIME, MAX_TIME)
    return (log_time - MIN_TIME) / (MAX_TIME - MIN_TIME)


def decode_events(events: Float[ndarray, "L"]) -> list[int]:
    return find_peaks(-events, height=0.6, distance=4)[0].tolist()


# == extents ==


def extents(
    regions: list[tuple[Real, Real]], frame_times: FrameTimes
) -> Float[ndarray, "L"]:
    """1 within extents, 0 everywhere else"""
    holds = np.zeros_like(frame_times)
    for s, e in regions:
        holds[(frame_times >= s) & (frame_times < e)] = 1
    return holds


def decode_extents(extents: Float[ndarray, "L"]) -> tuple[list[int], list[int]]:
    before_below = extents[:-1] <= 0
    after_below = extents[1:] <= 0

    start_idxs = sorted(np.argwhere(before_below & ~after_below)[:, 0].tolist())
    end_idxs = sorted(np.argwhere(~before_below & after_below)[:, 0].tolist())

    # ensure that start_idxs[i] < end_idxs[i] for all 0 <= i < min(len(start_idxs), len(end_idxs))
    cursor = 0
    for cursor, start in enumerate(start_idxs):
        try:
            while start >= end_idxs[cursor]:
                end_idxs.pop(cursor)
        except IndexError:
            break
    cursor += 1

    return start_idxs[:cursor], end_idxs[:cursor]


def slides(bm: Beatmap, frame_times: FrameTimes) -> Float[ndarray, "L"]:
    slides = np.zeros_like(frame_times)

    for ho in bm.hit_objects:
        if not isinstance(ho, Slider):
            continue

        for i in range(ho.slides):
            slide_start = ho.t + i * ho.slide_duration
            slide = (frame_times - slide_start) / ho.slide_duration
            if i % 2 == 1:
                slide = 1 - slide
            region = (slide >= 0) & (slide <= 1)
            slides[region] = slide[region]

    return slides


def decode_slides(slides: Float[ndarray, "L"]) -> list[int]:
    before_below = slides[:-1] <= 0
    after_below = slides[1:] <= 0

    fore_idxs = np.argwhere(before_below & ~after_below)[:, 0].tolist()
    back_idxs = np.argwhere(~before_below & after_below)[:, 0].tolist()

    return sorted([*fore_idxs, *back_idxs])


# == hit signal ==

HitEncoding = IntEnum(
    "HitEncoding",
    [
        "ONSET",
        "COMBO",
        "SLIDE",
        "SUSTAIN",
    ],
    start=0,
)
HIT_DIM = len(HitEncoding)

HitSignal = Float[ndarray, str(f"{HIT_DIM} L")]


def hit_signal(bm: Beatmap, frame_times: FrameTimes) -> HitSignal:
    """
    returns an array encoding a beatmap's hits:
    0. onsets
    1. new combos
    2. sustains (both sliders and spinners)
    3. the first slide of sliders
    """

    return (
        np.stack(
            [
                events([ho.t for ho in bm.hit_objects], frame_times),  # onsets
                events(
                    [ho.t for ho in bm.hit_objects if ho.new_combo], frame_times
                ),  # new combos
                slides(bm, frame_times),  # slides
                extents(
                    [
                        (ho.t, ho.end_time())
                        for ho in bm.hit_objects
                        if isinstance(ho, (Slider, Spinner))
                    ],
                    frame_times,
                ),  # sustains
            ]
        )
        * 2
        - 1
    )


Hit = Union[
    tuple[int, bool],  # hit(t, new_combo)
    tuple[int, bool, int, int],  # spin(t, new_combo, u, slides)
]

ONSET_TOL = 2


def decode_hit_signal(hit_signal: HitSignal) -> list[Hit]:
    onsets = hit_signal[HitEncoding.ONSET]
    onset_idxs = decode_events(onsets)

    # maps signal index to onset
    onset_idx_map = np.full_like(onsets, -1, dtype=int)
    for i, onset_idx in enumerate(onset_idxs):
        onset_idx_map[onset_idx - ONSET_TOL : onset_idx + ONSET_TOL + 1] = i

    new_combos = [False] * len(onset_idxs)
    for new_combo in decode_events(hit_signal[HitEncoding.COMBO]):
        onset_idx = onset_idx_map[new_combo]
        if onset_idx == -1:
            continue
        new_combos[onset_idx] = True

    sustain_ends = [-1] * len(onset_idxs)
    for sustain_start, sustain_end in zip(
        *decode_extents(hit_signal[HitEncoding.SUSTAIN])
    ):
        onset_idx = onset_idx_map[sustain_start]
        if onset_idx == -1:
            continue
        sustain_ends[onset_idx] = sustain_end

    slide_locs = decode_slides(hit_signal[HitEncoding.SLIDE])

    hits: list[Hit] = []
    for onset_loc, new_combo, sustain_end in zip(onset_idxs, new_combos, sustain_ends):
        hit = (onset_loc, new_combo)

        if sustain_end == -1 or sustain_end - onset_loc < 4:
            # sustain too short
            hits.append(hit)
            continue

        slides = len([loc for loc in slide_locs if onset_loc < loc < sustain_end])
        hits.append((*hit, sustain_end, slides))

    return hits
