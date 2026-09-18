import torch
from fractions import Fraction

# ComfyUI's native "VIDEO" type isn't a tensor - it's an object that wraps
# frames (+ optional audio + frame rate) and knows how to encode/mux itself
# to mp4/webm on demand. `VideoFromComponents` builds one of those objects
# directly from an IMAGE batch, which is exactly what we need here.
#
# The import path has moved around as ComfyUI's video API matured
# (comfy_api.input_impl / comfy_api.util -> comfy_api.latest._input_impl /
# comfy_api.latest._util). Try the modern location first, fall back to the
# older one, so this node keeps working across ComfyUI versions.
try:
    from comfy_api.latest import VideoFromComponents, VideoComponents
except ImportError:
    try:
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.util import VideoComponents
    except ImportError as e:
        raise ImportError(
            "VideoTailSlicer requires a ComfyUI version with the native "
            "video API (comfy_api.input_impl / comfy_api.util or "
            "comfy_api.latest). Please update ComfyUI."
        ) from e


class VideoTailSlicer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("IMAGE",),
                # Which end of the clip "seconds" is measured from.
                # "last_seconds"  -> keep the tail of the video (old behavior)
                # "first_seconds" -> keep the head of the video
                "slice_mode": (["last_seconds", "first_seconds"], {"default": "last_seconds"}),
                "seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            },
            "optional": {
                # Optional so existing graphs that only wire up IMAGE frames
                # keep working unchanged; hook this up if you want the VIDEO
                # output to carry sound too.
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "VIDEO")
    RETURN_NAMES = ("reference_video_clip", "first_frame_anchor", "last_frame_anchor", "video_output")
    FUNCTION = "slice_video"
    CATEGORY = "MiniMax Utils"

    def slice_video(self, video, slice_mode, seconds, fps, audio=None):
        # video shape is [Frames, Height, Width, Channels]
        total_frames = video.shape[0]
        frames_to_keep = int(seconds * fps)

        if slice_mode == "first_seconds":
            # Keep the head of the video: [0 : frames_to_keep]
            start_index = 0
            end_index = min(total_frames, frames_to_keep)
        else:
            # Keep the tail of the video (original behavior): [total-frames_to_keep : total]
            start_index = max(0, total_frames - frames_to_keep)
            end_index = total_frames

        # Create the reference clip (raw frames - kept as-is so anything
        # already wired to this output keeps working)
        reference_clip = video[start_index:end_index]

        # Anchor frames - always taken from the ORIGINAL, unsliced video, so
        # both are present regardless of which slice_mode is active.
        first_frame = video[0:1]   # Keeps dimensions as [1, H, W, C]
        last_frame = video[-1:]    # Keeps dimensions as [1, H, W, C]

        # --- Build a real, playable VIDEO output from the same sliced clip ---
        # frame_rate has to be a Fraction (not a raw float) for ComfyUI's
        # video encoder to read/write it correctly.
        frame_rate = Fraction(fps).limit_denominator(1000)

        sliced_audio = None
        if audio is not None:
            sliced_audio = self._slice_audio(audio, total_frames, start_index, end_index)

        video_output = VideoFromComponents(
            VideoComponents(
                images=reference_clip,
                audio=sliced_audio,
                frame_rate=frame_rate,
            )
        )

        return (reference_clip, first_frame, last_frame, video_output)

    @staticmethod
    def _slice_audio(audio, total_frames, start_index, end_index):
        """Trim the AUDIO dict's waveform to line up with the sliced frames,
        so the optional audio track stays in sync inside the VIDEO output,
        for either slice_mode (head or tail)."""
        waveform = audio["waveform"]  # shape [Batch, Channels, Samples]
        sample_rate = audio["sample_rate"]

        if total_frames <= 0:
            return audio

        total_samples = waveform.shape[-1]
        start_sample = int((start_index / total_frames) * total_samples)
        end_sample = int((end_index / total_frames) * total_samples)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))

        trimmed_waveform = waveform[..., start_sample:end_sample]

        return {"waveform": trimmed_waveform, "sample_rate": sample_rate}


class ImageBatchCount:
    """Pass-through utility: takes any IMAGE batch and reports how many
    frames are actually in it, as a real INT/STRING you can view with any
    display node (e.g. a native "Show Text" / int preview node).

    This does NOT recreate MiniMax H3 Easy's custom canvas link-badge system
    (that badge only exists for links plugged into that node's own "media"
    input and can't be attached to arbitrary sockets - see prior discussion).
    This node instead gives you the same information a different, reliable
    way: an actual widget/output value, not a canvas overlay, so it works on
    ANY IMAGE output from ANY node (reference_video_clip, first_frame_anchor,
    last_frame_anchor, etc.) - just wire the IMAGE in and read frame_count.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "frame_count", "info")
    FUNCTION = "count"
    CATEGORY = "MiniMax Utils"
    OUTPUT_NODE = True  # lets the count show up directly on the node itself too

    def count(self, image):
        # image shape is [Frames, Height, Width, Channels]
        frame_count = int(image.shape[0])
        height = int(image.shape[1])
        width = int(image.shape[2])
        info = f"{frame_count} frame{'s' if frame_count != 1 else ''} ({width}x{height})"

        return {
            "ui": {"text": [info]},
            "result": (image, frame_count, info),
        }


# --- THESE ARE THE MISSING LINES THAT FIX THE IMPORT ERROR ---
NODE_CLASS_MAPPINGS = {
    "VideoTailSlicer": VideoTailSlicer,
    "ImageBatchCount": ImageBatchCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTailSlicer": "Video Tail Slicer",
    "ImageBatchCount": "Image Batch Count",
}
