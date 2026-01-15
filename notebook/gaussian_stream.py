"""
Watch a folder for new scene captures and run the multi-object SAM3D pipeline.

Each scene is expected to live under ``input_root/<scene_id>/`` with:
  - ``image.png``: RGB source frame.
  - ``0.png, 1.png, ...``: binary masks (one per object), same resolution.

Any scene folder created after the watcher starts (or explicitly allowed via
``--process-existing``) will be picked up, processed, and written to
``output_root/<scene_id>.ply``. Optionally, a GIF render can be emitted too.
"""

import argparse
import logging
import os
import time
import math
from typing import Dict, Set

import imageio
import torch

# Match notebook CUDA/extension settings to reuse compiled artifacts.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"
os.environ.setdefault("FORCE_CUDA", "1")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# Keep sam3d_objects init lightweight (matches notebook behavior).
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

# Make notebook modules importable when run from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in os.sys.path:
    os.sys.path.insert(0, THIS_DIR)
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR", os.path.join(REPO_ROOT, ".torch_extensions")
)
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
# Help mitigate fragmentation for large attention ops.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
    Gaussian,
)


def gaussian_to_device(gaussian: Gaussian, device: str) -> Gaussian:
    """
    Move all gaussian tensors/buffers to a given device.
    """
    gaussian.device = device
    gaussian.aabb = gaussian.aabb.to(device)
    gaussian.scale_bias = gaussian.scale_bias.to(device)
    gaussian.rots_bias = gaussian.rots_bias.to(device)
    gaussian.opacity_bias = gaussian.opacity_bias.to(device)
    for attr in [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ]:
        tensor = getattr(gaussian, attr, None)
        if tensor is not None:
            setattr(gaussian, attr, tensor.to(device))
    return gaussian

from notebook.inference import (  # noqa: E402
    Inference,
    load_image,
    load_masks,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)


LOGGER = logging.getLogger("gaussian_stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "checkpoints", "hf", "pipeline.yaml"),
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join(REPO_ROOT, "snapshots"),
        help="Root directory where new scene folders appear.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "notebook", "gaussians", "multi"),
        help="Directory to store output PLY (and optional GIF).",
    )
    parser.add_argument(
        "--mask-extension",
        default=".png",
        help="Extension used for mask files (e.g., .png or .jpg).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between scans for new scenes.",
    )
    parser.add_argument(
        "--stabilize-seconds",
        type=float,
        default=2.0,
        help="Wait this long after last scene file modification before processing.",
    )
    parser.add_argument(
        "--render-gif",
        action="store_true",
        help="Also render a GIF orbit of the scene.",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="If set, process scenes already present at startup (default: only new).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to the inference pipeline.",
    )
    return parser.parse_args()


def scene_ready(scene_dir: str, mask_ext: str, stabilize_seconds: float) -> bool:
    """
    Check that a scene directory looks complete:
    - image.png exists
    - at least one mask with contiguous numbering from 0
    - files have been stable for the requested duration
    """
    image_path = os.path.join(scene_dir, "image.png")
    if not os.path.exists(image_path):
        return False

    mask_idx = 0
    has_mask = False
    while True:
        mask_path = os.path.join(scene_dir, f"{mask_idx}{mask_ext}")
        if os.path.exists(mask_path):
            has_mask = True
            mask_idx += 1
        else:
            break

    if not has_mask:
        return False

    latest_mtime = max(
        os.path.getmtime(os.path.join(scene_dir, fname))
        for fname in os.listdir(scene_dir)
    )
    return (time.time() - latest_mtime) >= stabilize_seconds


def discover_scenes(
    root: str, since: float, processed: Set[str], mask_ext: str, stabilize_seconds: float
) -> Dict[str, str]:
    """
    Return mapping of scene_id -> scene_path for scenes created after ``since``
    (unless already processed) that appear ready for processing.
    """
    scenes = {}
    if not os.path.isdir(root):
        LOGGER.debug("Input root %s does not exist or is not a directory", root)
        return scenes

    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        scene_id = entry.name
        if scene_id in processed:
            LOGGER.debug("Skipping already processed scene %s", scene_id)
            continue
        mtime = entry.stat().st_mtime
        if mtime < since:
            LOGGER.debug(
                "Skipping scene %s (mtime %.2f before start %.2f)", scene_id, mtime, since
            )
            continue
        scene_dir = entry.path
        if scene_ready(scene_dir, mask_ext, stabilize_seconds):
            scenes[scene_id] = scene_dir
            LOGGER.info("Scene %s is ready", scene_id)
        else:
            LOGGER.debug("Scene %s not ready yet", scene_id)
    return scenes


def run_inference_on_scene(
    inference: Inference,
    scene_id: str,
    scene_dir: str,
    output_root: str,
    mask_ext: str,
    seed: int,
    render_gif: bool,
):
    image_path = os.path.join(scene_dir, "image.png")
    LOGGER.info("Processing scene %s", scene_id)
    image = load_image(image_path)
    masks = load_masks(scene_dir, extension=mask_ext)
    total_masks = len(masks)
    LOGGER.info("Scene %s has %d mask(s)", scene_id, total_masks)

    base_stage1_steps = 30
    drop_stage1_steps = 25
    drop_threshold = 20

    def scaled_steps(mask_count: int) -> int:
        # Up to threshold masks: keep base steps. Beyond: drop to lower value.
        return base_stage1_steps if mask_count <= drop_threshold else drop_stage1_steps

    stage1_steps = scaled_steps(total_masks)
    LOGGER.info(
        "Using stage1_inference_steps=%d (base=%d, drop=%d if masks>%d)",
        stage1_steps,
        base_stage1_steps,
        drop_stage1_steps,
        drop_threshold,
    )

    outputs = []
    for idx, mask in enumerate(masks):
        LOGGER.info("Running inference for mask %d/%d", idx + 1, total_masks)
        out = inference(image, mask, seed=seed, stage1_inference_steps=stage1_steps)
        # Move heavy tensors to CPU to save GPU memory before merging.
        g = out["gaussian"][0]
        gaussian_to_device(g, "cpu")
        out["rotation"] = out["rotation"].cpu()
        out["translation"] = out["translation"].cpu()
        out["scale"] = out["scale"].cpu()
        outputs.append(out)
        LOGGER.info("Finished mask %d/%d", idx + 1, total_masks)

    scene_gs = make_scene(*outputs)
    # Move merged gaussian back to GPU for rendering.
    gaussian_to_device(scene_gs, "cuda")
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    os.makedirs(output_root, exist_ok=True)
    ply_path = os.path.join(output_root, f"{scene_id}.ply")
    scene_gs.save_ply(ply_path)
    LOGGER.info("Saved PLY to %s", ply_path)

    if render_gif:
        video = render_video(scene_gs, r=1, fov=60, resolution=512)["color"]
        gif_path = os.path.join(output_root, f"{scene_id}.gif")
        imageio.mimsave(
            gif_path,
            video,
            format="GIF",
            duration=1000 / 30,
            loop=0,
        )
        LOGGER.info("Saved GIF to %s", gif_path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start_time = time.time()
    if args.process_existing:
        # Start far in the past to pick up all existing scenes.
        start_time = 0.0

    LOGGER.info("Loading pipeline from %s", args.config)
    inference = Inference(args.config, compile=False)

    processed: Set[str] = set()
    LOGGER.info(
        "Watching %s for new scenes (mask ext=%s). Output to %s",
        args.input_root,
        args.mask_extension,
        args.output_root,
    )
    while True:
        try:
            scenes = discover_scenes(
                args.input_root,
                start_time,
                processed,
                args.mask_extension,
                args.stabilize_seconds,
            )
            for scene_id, scene_dir in scenes.items():
                try:
                    run_inference_on_scene(
                        inference,
                        scene_id,
                        scene_dir,
                        args.output_root,
                        args.mask_extension,
                        args.seed,
                        args.render_gif,
                    )
                    processed.add(scene_id)
                    torch.cuda.empty_cache()
                    LOGGER.info("Finished scene %s; freed CUDA cache", scene_id)
                except Exception:
                    LOGGER.exception("Failed processing scene %s", scene_id)
                    torch.cuda.empty_cache()

            if not scenes:
                LOGGER.info("No new ready scenes; sleeping %.1fs", args.poll_interval)
            time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            LOGGER.info("Exiting on keyboard interrupt.")
            break


if __name__ == "__main__":
    main()
