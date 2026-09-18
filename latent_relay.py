"""Save and reload a MiniMax H3 AV latent between runs.

Chaining clips through video costs a decode and a re-encode at every join, and
that round trip is not neutral: it darkens by roughly 2.4% each time and the
error compounds down a chain. Chaining latents avoids it, but a latent normally
only exists inside one execution, which forces every clip into a single
graph -- slow, and impossible to review clip by clip.

These two nodes remove that constraint. Save the latent after a render, review
the video, then load the latent back on the next run and continue from it. The
frames handed to the model are the exact numbers the model produced, so the
join is lossless while the workflow stays one clip at a time.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch

import comfy.utils
import folder_paths

from . import nodes as h3_nodes


_SUBDIR = "h3_latents"
_SUFFIX = ".h3latent.safetensors"


def _relay_dir() -> str:
    path = os.path.join(folder_paths.get_output_directory(), _SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _list_saved() -> list[str]:
    """Every saved latent, newest first.

    Plain ``.safetensors`` is accepted as well as the full suffix so a file can
    be renamed freely in a file manager without disappearing from the list.
    """
    try:
        names = [
            f for f in os.listdir(_relay_dir())
            if f.endswith(_SUFFIX) or f.endswith(".safetensors")
        ]
    except OSError:
        return []
    names.sort(key=lambda f: os.path.getmtime(os.path.join(_relay_dir(), f)), reverse=True)
    return names


def _streams(value: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pull the video and audio streams out of segments or a plain AV latent."""
    samples = getattr(value, "samples", None)
    if samples:
        last = samples[-1]
        return getattr(last, "video_latent", None), getattr(last, "audio_latent", None)
    if isinstance(value, Mapping) and value.get("samples") is not None:
        return h3_nodes._segment_latent_streams(value)
    raise ValueError("Expected a MiniMax H3 segment result or an AV latent")


class MiniMaxH3EasySaveLatent:
    """Write a rendered clip's AV latent to disk for the next run to continue from."""

    CATEGORY = "MiniMaxH3-Easy"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save a clip's AV latent so the next run can continue from it without going "
        "through video. Chaining through video costs a decode and a re-encode at every "
        "join, which darkens the picture a little each time and compounds down a chain. "
        "Saving the latent keeps the join lossless while still letting you render and "
        "review one clip at a time."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "clip",
                    "tooltip": "Name for the saved file, under output/h3_latents as "
                               "<name>_00001.h3latent.safetensors. Use something descriptive per "
                               "scene, such as sato_interrogation, and the numbering keeps each "
                               "scene's clips together.\n\n"
                               "Files can also be renamed afterwards in a file manager; keep the "
                               ".safetensors ending and they stay in the list. The dropdown in "
                               "Load H3 Latent is sorted newest first.",
                }),
            },
            "optional": {
                "segments": (h3_nodes.SEGMENT_RESULT_TYPE,),
                "latent": ("LATENT",),
            },
        }

    def save(self, filename_prefix="clip", segments=None, latent=None):
        source = segments if segments is not None else latent
        if source is None:
            raise ValueError("Connect either segments or latent to save")
        video, audio = _streams(source)
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("No usable video latent to save")

        # Spaces become underscores so a descriptive name survives intact;
        # anything else that could escape the folder is dropped.
        cleaned = str(filename_prefix).strip().replace(" ", "_")
        prefix = "".join(c for c in cleaned if c.isalnum() or c in "-_") or "clip"
        directory = _relay_dir()
        index = 1
        while os.path.exists(os.path.join(directory, f"{prefix}_{index:05d}{_SUFFIX}")):
            index += 1
        path = os.path.join(directory, f"{prefix}_{index:05d}{_SUFFIX}")

        tensors = {"video": video.detach().to("cpu").contiguous()}
        if isinstance(audio, torch.Tensor):
            tensors["audio"] = audio.detach().to("cpu").contiguous()
        comfy.utils.save_torch_file(tensors, path, metadata={
            "format": "minimax_h3_av_latent_v1",
            "video_shape": str(tuple(video.shape)),
        })
        frames = int(video.shape[2])
        print(
            f"[MiniMax H3 Easy] Saved AV latent: {os.path.basename(path)} "
            f"({frames} video tokens, {'with' if 'audio' in tensors else 'no'} audio). "
            f"Load it into the next run's seed_latent to continue with no quality loss."
        )
        return {"ui": {"text": [os.path.basename(path)]}, "result": (path,)}


class MiniMaxH3EasyLoadLatent:
    """Load a saved AV latent to continue from in a later run."""

    CATEGORY = "MiniMaxH3-Easy"
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Load a latent saved by Save H3 Latent and feed it to Context Segments' "
        "seed_latent input. The continuation starts from the exact numbers the previous "
        "clip was made of, with no decode and no re-encode, so no colour or contrast "
        "shift at the join."
    )

    @classmethod
    def INPUT_TYPES(cls):
        saved = _list_saved()
        return {
            "required": {
                "latent_file": (saved or ["(none saved yet)"], {
                    "tooltip": "Newest first. Produced by Save H3 Latent, under "
                               "output/h3_latents.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, latent_file, **kwargs):
        path = os.path.join(_relay_dir(), str(latent_file))
        return os.path.getmtime(path) if os.path.exists(path) else float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, latent_file, **kwargs):
        if str(latent_file).startswith("("):
            return "No saved latents yet. Run Save H3 Latent first."
        if not os.path.exists(os.path.join(_relay_dir(), str(latent_file))):
            return f"Saved latent not found: {latent_file}"
        return True

    def load(self, latent_file):
        path = os.path.join(_relay_dir(), str(latent_file))
        if not os.path.exists(path):
            raise ValueError(f"Saved latent not found: {latent_file}")
        tensors = comfy.utils.load_torch_file(path, safe_load=True)
        video = tensors.get("video")
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(f"{latent_file} does not contain a MiniMax H3 video latent")
        audio = tensors.get("audio")
        if not isinstance(audio, torch.Tensor):
            # The seed path only needs audio when the shot generates its own sound;
            # an empty stream keeps the packing valid without inventing content.
            audio = torch.zeros(
                (int(video.shape[0]), 8, 4, 1), dtype=video.dtype
            )
        print(
            f"[MiniMax H3 Easy] Loaded AV latent {latent_file} "
            f"({int(video.shape[2])} video tokens) for a lossless continuation."
        )
        return (h3_nodes._segment_pack_latent(video, audio),)
