"""Automatic overlap trimming for MiniMax H3 continuation clips.

Native Guide renders the first ``context_frames`` of a continuation as a
re-render of the source video's tail. Stitching therefore means cutting the
source at the point that re-render began. A fixed cut at ``context_frames`` is
close but not reliable: the guide's coverage is quantised to latent tokens, and
the re-render drifts slightly from the original. So this searches for the cut
instead of assuming it -- it finds where the source best matches the
continuation's opening and cuts there.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from . import nodes as h3_nodes


def _match_features(frames: torch.Tensor, size: int = 48) -> torch.Tensor:
    """Small grayscale signature per frame, for fast and robust comparison.

    Downsampling suppresses the codec noise and fine re-render differences that
    would otherwise dominate a raw pixel distance, while keeping composition and
    subject position -- which is what alignment actually depends on.
    """
    value = frames.permute(0, 3, 1, 2).float()
    if value.shape[1] >= 3:
        # Rec.601 luma. Cheaper than colour matching and immune to slight
        # colour drift between the source and its re-render.
        weights = torch.tensor([0.299, 0.587, 0.114], device=value.device).view(1, 3, 1, 1)
        value = (value[:, :3] * weights).sum(dim=1, keepdim=True)
    else:
        value = value[:, :1]
    value = torch.nn.functional.interpolate(
        value, size=(size, size), mode="area"
    ).flatten(1)
    # Per-frame standardisation, so exposure differences do not shift the match.
    value = value - value.mean(dim=1, keepdim=True)
    norm = value.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return value / norm


def find_cut_frame(
    source: torch.Tensor,
    continuation: torch.Tensor,
    search_frames: int,
    match_window: int,
) -> tuple[int, float]:
    """Frame in ``source`` where ``continuation`` begins.

    Returns ``(cut_frame, similarity)``. ``cut_frame`` is the first source frame
    to discard; keeping ``source[:cut_frame]`` and appending the continuation
    whole gives a continuous join. Similarity is cosine, 1.0 being identical.
    """
    total = int(source.shape[0])
    window = max(1, min(int(match_window), int(continuation.shape[0]), total))
    latest = total - window
    if latest < 0:
        return total, 0.0
    earliest = max(0, latest - max(0, int(search_frames)))

    target = _match_features(continuation[:window])
    best_frame, best_score = latest, float("-inf")
    for start in range(earliest, latest + 1):
        candidate = _match_features(source[start:start + window])
        score = float((candidate * target).sum(dim=1).mean())
        if score > best_score:
            best_score, best_frame = score, start
    return best_frame, best_score


def _trim_audio(audio: Any, keep_frames: int, fps: float) -> Any:
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        return None
    rate = int(h3_nodes._audio_sample_rate(audio))
    keep = max(0, int(round(float(keep_frames) / max(1e-6, float(fps)) * rate)))
    keep = min(keep, int(waveform.shape[-1]))
    return {"waveform": waveform[..., :keep].contiguous(), "sample_rate": rate}


def _audio_tail_from(audio: Any, skip_frames: int, fps: float) -> Any:
    """Drop the first ``skip_frames`` worth of audio, keeping the rest."""
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        return None
    rate = int(h3_nodes._audio_sample_rate(audio))
    skip = max(0, int(round(float(skip_frames) / max(1e-6, float(fps)) * rate)))
    skip = min(skip, int(waveform.shape[-1]))
    return {"waveform": waveform[..., skip:].contiguous(), "sample_rate": rate}


def _crossfade_join(first: Any, second: Any, millis: float = 12.0) -> Any:
    """Concatenate two tracks with a short fade at the seam, preserving length.

    Splicing two unrelated waveforms at an arbitrary sample leaves a step
    discontinuity, which is audible as a click. This fades the tail of the first
    track down and the head of the second up across a few milliseconds, then
    concatenates.

    The fades are applied in place rather than overlapping the two tracks,
    so the result is exactly as long as its inputs. A true crossfade would
    consume its overlap and shorten the audio by a few milliseconds at every
    join, and across a chain of clips that accumulates into audible A/V drift.
    Keeping sync matters more here than the very slight dip in level.
    """
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return _concat_audio(first, second)
    a, b = first.get("waveform"), second.get("waveform")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _concat_audio(first, second)
    rate = int(h3_nodes._audio_sample_rate(first))
    n = int(round(max(0.0, float(millis)) / 1000.0 * rate))
    n = min(n, int(a.shape[-1]), int(b.shape[-1]))
    if n < 8:
        return _concat_audio(first, second)
    channels = max(int(a.shape[1]), int(b.shape[1]))
    if int(a.shape[1]) < channels:
        a = a.repeat(1, channels // int(a.shape[1]), 1)
    if int(b.shape[1]) < channels:
        b = b.repeat(1, channels // int(b.shape[1]), 1)
    a, b = a.clone(), b.clone()
    ramp = torch.linspace(0.0, 1.0, n, dtype=a.dtype, device=a.device)
    a[..., -n:] = a[..., -n:] * torch.cos(ramp * 1.5707963)
    b[..., :n] = b[..., :n] * torch.sin(ramp * 1.5707963)
    return {"waveform": torch.cat([a, b], dim=-1).contiguous(), "sample_rate": rate}


def _concat_audio(first: Any, second: Any) -> Any:
    tracks = [a for a in (first, second)
              if isinstance(a, Mapping) and isinstance(a.get("waveform"), torch.Tensor)]
    if not tracks:
        return None
    if len(tracks) == 1:
        return tracks[0]
    rate = int(h3_nodes._audio_sample_rate(tracks[0]))
    parts = []
    for track in tracks:
        wave = track["waveform"]
        if int(h3_nodes._audio_sample_rate(track)) != rate:
            wave = torch.nn.functional.interpolate(
                wave, scale_factor=rate / float(h3_nodes._audio_sample_rate(track)),
                mode="linear", align_corners=False,
            )
        parts.append(wave)
    channels = max(int(p.shape[1]) for p in parts)
    parts = [p.repeat(1, channels // int(p.shape[1]), 1) if int(p.shape[1]) < channels else p
             for p in parts]
    return {"waveform": torch.cat(parts, dim=-1).contiguous(), "sample_rate": rate}


class MiniMaxH3EasyStitchContinuation:
    """Trim the re-rendered overlap and join a continuation onto its source."""

    CATEGORY = "MiniMaxH3-Easy"
    FUNCTION = "stitch"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "cut_frame", "report")
    DESCRIPTION = (
        "Finds where a Native Guide continuation's re-rendered opening matches the "
        "end of its source video, cuts the source there, and joins the two. Use "
        "auto alignment unless you need a specific cut."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("VIDEO",),
                "continuation_video": ("VIDEO",),
                "alignment": (["auto", "fixed overlap"], {
                    "default": "auto",
                    "tooltip": "auto searches for the best matching cut. 'fixed overlap' "
                               "cuts exactly overlap_frames from the end of the source.",
                }),
                "overlap_frames": ("INT", {
                    "default": 39, "min": 0, "max": 600,
                    "tooltip": "Your context_frames value. Used directly in 'fixed overlap', "
                               "and as the centre of the search in auto.",
                }),
                "search_frames": ("INT", {
                    "default": 30, "min": 0, "max": 300,
                    "tooltip": "How far either side of overlap_frames auto may look.",
                }),
                "match_window": ("INT", {
                    "default": 8, "min": 1, "max": 60,
                    "tooltip": "Frames compared at each candidate cut. More is steadier on "
                               "near-static shots, less is sharper on fast motion.",
                }),
                "window_audio": (["from source", "from continuation"], {
                    "default": "from source",
                    "tooltip": "Where the audio for the overlap comes from. The continuation "
                               "re-renders footage you already have, and it re-renders the sound "
                               "too -- usually as invented, unintelligible speech, because there "
                               "is no partially-correct voice. 'from source' replaces that span "
                               "with the original audio, so a line being spoken across the join "
                               "carries over intact. Switch to 'from continuation' only if the "
                               "re-rendered mouth movement does not match the original.",
                }),
                "color_match": (["off", "mean", "mean + contrast"], {
                    "default": "mean + contrast",
                    "tooltip": "Correct colour drift in the continuation before joining. Each "
                               "generation re-encodes the previous video, and small shifts in "
                               "brightness and contrast compound every time you chain a clip. "
                               "This measures the difference across the overlap, where both "
                               "videos show the same moment, and corrects the whole continuation "
                               "to match the source. 'mean + contrast' fixes both brightness and "
                               "contrast; 'mean' fixes brightness only.",
                }),
            },
            "optional": {
                "h3_context": ("MINIMAX_H3_CONTEXT", {
                    "tooltip": "Connect Context Segments' H3 Context here and overlap_frames is "
                               "read from it automatically, so it always matches the continuation "
                               "window actually used. The widget is then ignored.",
                }),
            },
        }

    @staticmethod
    def _overlap_from_context(h3_context, fallback: int) -> tuple[int, bool]:
        """Read the continuation window straight off the Context Segments plan."""
        plan = getattr(h3_context, "segment_plan", None)
        if not isinstance(plan, Mapping):
            return int(fallback), False
        seed = plan.get("seed_context")
        if isinstance(seed, Mapping):
            frames = seed.get("lock_frames")
            if isinstance(frames, int) and frames > 0:
                return int(frames), True
        frames = plan.get("context_length")
        if isinstance(frames, int) and frames > 0:
            return int(frames), True
        return int(fallback), False

    def stitch(
        self, source_video, continuation_video, alignment,
        overlap_frames, search_frames, match_window,
        color_match="mean + contrast", window_audio="from source", h3_context=None,
    ):
        overlap_frames, from_context = self._overlap_from_context(h3_context, overlap_frames)
        source, source_audio, source_fps = h3_nodes._video_parts(source_video)
        follow, follow_audio, _follow_fps = h3_nodes._video_parts(continuation_video)
        source = h3_nodes._normalize_video_frames(source)
        follow = h3_nodes._normalize_video_frames(follow)
        fps = float(source_fps or h3_nodes.h3.FPS)

        if int(source.shape[0]) == 0 or int(follow.shape[0]) == 0:
            raise ValueError("Stitch Continuation needs two non-empty videos")

        # Match geometry before comparing or joining. Testing at 360p against a
        # 480p source is a normal case, so resize rather than refusing.
        resized = False
        if follow.shape[1:3] != source.shape[1:3]:
            follow = h3_nodes.h3._resize(
                follow, int(source.shape[2]), int(source.shape[1]), "center"
            )
            resized = True

        total = int(source.shape[0])
        if str(alignment) == "fixed overlap":
            cut = max(0, total - int(overlap_frames))
            score = float("nan")
        else:
            nominal = max(0, total - int(overlap_frames))
            span = max(0, int(search_frames))
            window = max(1, min(int(match_window), int(follow.shape[0])))
            upper = min(total - window, nominal + span)
            lower = max(0, nominal - span)
            if upper < lower:
                cut, score = max(0, min(nominal, total)), float("nan")
            else:
                cut, score = find_cut_frame(
                    source[:upper + window], follow, upper - lower, window
                )

        cut = max(0, min(int(cut), total))

        # Correct drift using the overlap, where both clips show the same
        # moment, so any difference between them is drift rather than content.
        matched = "off"
        if str(color_match) != "off" and cut < total:
            window = min(int(match_window), total - cut, int(follow.shape[0]))
            if window >= 1:
                ref = source[cut:cut + window]
                cur = follow[:window]
                ref_mean = ref.mean(dim=(0, 1, 2), keepdim=True)
                cur_mean = cur.mean(dim=(0, 1, 2), keepdim=True)
                if str(color_match) == "mean + contrast":
                    ref_std = ref.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
                    cur_std = cur.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
                    scale = (ref_std / cur_std).clamp(0.5, 2.0)
                else:
                    scale = torch.ones_like(ref_mean)
                follow = ((follow - cur_mean) * scale + ref_mean).clamp(0.0, 1.0)
                matched = f"{color_match} (gain {float(scale.mean()):.3f})"

        joined = torch.cat([source[:cut], follow], dim=0).contiguous()

        # The overlap is the continuation re-rendering footage the source
        # already contains, so the source's real audio is the correct audio for
        # that span. Taking it from the source keeps a line spoken across the
        # join intact instead of replacing it with invented speech.
        overlap_used = min(int(total - cut), int(follow.shape[0]))
        audio_source = "continuation"
        if str(window_audio) == "from source" and overlap_used > 0 and source_audio is not None:
            head = _trim_audio(source_audio, cut + overlap_used, fps)
            tail = _audio_tail_from(follow_audio, overlap_used, fps)
            audio = _crossfade_join(head, tail) if tail is not None else head
            audio_source = f"source through the {overlap_used}-frame overlap, then continuation"
        else:
            audio = _concat_audio(_trim_audio(source_audio, cut, fps), follow_audio)

        # cut == 0 means the whole source was treated as overlap and dropped.
        # That happens legitimately only on a degenerate input; far more often
        # it means overlap_frames exceeds the source length, in which case the
        # node has silently thrown away every frame the user supplied.
        dropped_all = cut == 0 and total > 0
        report = (
            f"source {total}f + continuation {int(follow.shape[0])}f -> {int(joined.shape[0])}f\n"
            f"cut source at frame {cut} (dropped {total - cut} overlapping frames)\n"
            f"overlap_frames: {int(overlap_frames)}"
            + (" (read from h3_context)" if from_context else " (widget)") + "\n"
            + f"colour match: {matched}\n"
            + f"overlap audio: {audio_source}\n"
            + f"alignment: {alignment}"
            + ("" if score != score else f", match {score:.4f}")
            + (f"\ncontinuation resized to {int(source.shape[2])}x{int(source.shape[1])}"
               if resized else "")
            + ("\nWARNING: the entire source video was dropped. overlap_frames "
               f"({int(overlap_frames)}) is >= the source length ({total}); lower it "
               "or supply a longer source." if dropped_all else "")
        )
        return (joined, audio, fps, int(cut), report)
