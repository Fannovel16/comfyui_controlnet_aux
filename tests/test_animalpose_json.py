"""Regression tests for issue #627: "AssertionError" on Render Pose JSON (Animal).

The AP10K animal encoder used to emit nested ``K x 3`` lists, while
``decode_json_as_poses`` expects the flat OpenPose layout ``[x, y, c, ...]``
with binary confidence flags -- the same layout the human encoder
(``encode_poses_as_dict`` / ``compress_keypoints``) produces. These tests pin
the fixed format and guard the unaffected paths.

Runs on CPU only and does not download any model.

Note: pytest >= 8.4 currently fails to collect any test in this repository
regardless of content (the repo-root ``__init__.py`` ComfyUI entry point gets
imported as a top-level module during collection and its relative import
fails). These tests were verified with pytest 7.4.4.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from custom_controlnet_aux.dwpose import (
    decode_json_as_poses,
    draw_animalposes,
    encode_poses_as_dict,
)
from custom_controlnet_aux.dwpose.animalpose import encode_animal_keypoints
from custom_controlnet_aux.dwpose.body import Keypoint
from custom_controlnet_aux.dwpose.types import BodyResult, PoseResult

K = 17  # AP10K keypoint count
H = W = 512


def make_animal_scores(seed=0, zero_index=None):
    """Build one animal's (K, 3) array the way AnimalPoseImage.__call__ does:
    coordinates plus SimCC scores clamped to [0, 1] (fractional in practice)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, W, size=(K, 2)).round(1)
    score = rng.uniform(0.3, 1.0, size=(K, 1)).round(3)
    if zero_index is not None:
        score[zero_index, 0] = 0.0
    return np.concatenate([xy, score], axis=-1)


def test_encode_animal_keypoints_flattens_and_binarizes():
    animal = make_animal_scores()
    flat = encode_animal_keypoints(animal)
    assert len(flat) == K * 3
    confidences = flat[2::3]
    assert set(confidences) <= {1.0, 0.0}
    for i, (x, y, _) in enumerate(animal):
        assert flat[3 * i] == float(x)
        assert flat[3 * i + 1] == float(y)


def test_animal_json_roundtrip_through_decoder():
    animal = make_animal_scores(zero_index=5)
    pose_json = {
        "version": "ap10k",
        "animals": [encode_animal_keypoints(animal)],
        "canvas_height": H,
        "canvas_width": W,
    }
    # SavePoseKpsAsJsonFile-style round trip: json.dump -> file -> json.load
    pose_json = json.loads(json.dumps(pose_json))
    _, animals, h, w = decode_json_as_poses(pose_json)
    assert (h, w) == (H, W)
    assert len(animals) == 1
    kps = animals[0]
    assert sum(kp is not None for kp in kps) == K - 1
    assert kps[5] is None  # score 0 -> absent keypoint, must not crash
    img = draw_animalposes(animals, h, w)
    assert img.shape == (H, W, 3)
    assert img.any()  # bones actually drawn


def test_no_animals_path_unchanged():
    _, animals, h, w = decode_json_as_poses(
        {"version": "ap10k", "animals": [], "canvas_height": H, "canvas_width": W}
    )
    assert animals == []
    assert (h, w) == (H, W)


def test_human_path_roundtrip_unchanged():
    kps = [Keypoint(x=10.0 * i, y=20.0 * i) for i in range(18)]
    pose = PoseResult(body=BodyResult(keypoints=kps), left_hand=None, right_hand=None, face=None)
    poses_json = encode_poses_as_dict([pose], H, W)
    human_poses, animals, h, w = decode_json_as_poses(poses_json)
    assert len(human_poses) == 1
    assert animals == []
    assert sum(kp is not None for kp in human_poses[0].body.keypoints) == 18


def test_nested_legacy_format_still_asserts():
    # Documents the pre-fix behavior from #627: a nested K x 3 list fails the
    # flat-layout assert in decompress_keypoints (17 % 3 == 2).
    animal = make_animal_scores()
    with pytest.raises(AssertionError):
        decode_json_as_poses(
            {
                "version": "ap10k",
                "animals": [animal.tolist()],
                "canvas_height": H,
                "canvas_width": W,
            }
        )
