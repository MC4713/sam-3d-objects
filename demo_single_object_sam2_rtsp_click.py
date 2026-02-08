#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 live RTSP click/box prompting.

Hotkeys in live window:
  p = segment current frame with point prompts
  b = segment current frame with box prompt
  m = segment current frame with mixed prompts (cv2 only)
  q = quit
"""

import argparse
import json
import os
import queue
import re
import threading
from datetime import datetime

import numpy as np
from PIL import Image

from demo_single_object_sam2_click import (
    build_output_dirs,
    get_box,
    get_points,
    get_prompts_cv2_mixed,
    overlay_mask,
    save_mask,
    setup_sam2_backend,
)


def _sanitize_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "stream"


def _source_tag(source: str) -> str:
    if "//" in source:
        source = source.split("//", 1)[-1]
    return _sanitize_name(source.replace("/", "_"))


def _predict_masks(
    image_rgb,
    backend,
    backend_obj,
    mode,
    objects,
    n_pos,
    n_neg,
    multimask,
):
    import torch

    all_point_coords = []
    all_point_labels = []
    all_boxes = []
    all_modes = []

    if mode == "mixed":
        all_point_coords, all_point_labels, all_boxes, all_modes = get_prompts_cv2_mixed(
            image_rgb, objects, n_pos, n_neg
        )
    elif mode == "points":
        for i in range(objects):
            pts, labels = get_points(image_rgb, n_pos, n_neg, force_mpl=False)
            if len(pts) == 0:
                raise RuntimeError(f"No points selected for object {i + 1}.")
            all_point_coords.append(np.array(pts, dtype=np.float32))
            all_point_labels.append(np.array(labels, dtype=np.int32))
            all_boxes.append(None)
            all_modes.append("points")
    elif mode == "box":
        for i in range(objects):
            box = get_box(image_rgb, force_mpl=False)
            if box is None:
                raise RuntimeError(f"No box selected for object {i + 1}.")
            all_boxes.append(np.array(box, dtype=np.float32))
            all_point_coords.append(None)
            all_point_labels.append(None)
            all_modes.append("box")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    masks_out = []
    if backend == "sam2_repo":
        sam2_predictor = backend_obj
        sam2_predictor.set_image(np.array(image_rgb))
        for i in range(objects):
            masks, scores, _ = sam2_predictor.predict(
                point_coords=all_point_coords[i],
                point_labels=all_point_labels[i],
                box=all_boxes[i],
                multimask_output=multimask,
            )
            mask_idx = int(np.argmax(scores)) if scores is not None else 0
            masks_out.append(masks[mask_idx].astype(bool))
    else:
        if mode == "mixed":
            raise RuntimeError("Transformers backend does not support mixed mode in this script.")

        model, processor = backend_obj
        device = next(model.parameters()).device

        input_points = None
        input_labels = None
        input_boxes = None
        if mode == "points":
            input_points = [[pts.tolist() for pts in all_point_coords]]
            input_labels = [[labs.tolist() for labs in all_point_labels]]
        if mode == "box":
            input_boxes = [[
                [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                for b in all_boxes
            ]]

        inputs = processor(
            images=Image.fromarray(image_rgb),
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=multimask)

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        if masks.ndim == 3:
            masks_by_obj = masks[None]
        elif masks.ndim == 4:
            masks_by_obj = masks
        elif masks.ndim == 5 and masks.shape[0] == 1:
            masks_by_obj = masks[0]
        else:
            raise RuntimeError(f"Unexpected masks shape: {masks.shape}")

        scores = None
        if hasattr(outputs, "iou_scores"):
            scores = outputs.iou_scores.cpu().numpy()
        elif hasattr(outputs, "pred_iou"):
            scores = outputs.pred_iou.cpu().numpy()

        for obj_idx in range(masks_by_obj.shape[0]):
            obj_masks = masks_by_obj[obj_idx]
            if scores is not None:
                if scores.ndim == 3:
                    obj_scores = scores[0, obj_idx]
                elif scores.ndim == 2:
                    obj_scores = scores[obj_idx]
                else:
                    obj_scores = scores
                mask_idx = int(np.argmax(obj_scores))
            else:
                mask_idx = 0
            masks_out.append(obj_masks[mask_idx].astype(bool))

    return masks_out, all_modes


def _save_run(frame_bgr, source, masks_out, modes, backend_name, mode_name, args):
    frame_rgb = frame_bgr[:, :, ::-1].copy()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_stub = f"{_source_tag(source)}_{stamp}.png"

    out_root, masks_dir, overlays_dir, logs_dir = build_output_dirs(image_stub, root_dir=args.out_root)

    frame_path = os.path.join(out_root, "frame.png")
    Image.fromarray(frame_rgb).save(frame_path)

    mask_paths = []
    for i, mask in enumerate(masks_out):
        mask_path = os.path.join(masks_dir, f"{i}.png")
        mask_paths.append(save_mask(mask, image_stub, out_path=mask_path))

        overlay = overlay_mask(frame_rgb, mask)
        overlay_path = os.path.join(overlays_dir, f"{i}.png")
        Image.fromarray(overlay).save(overlay_path)

    log_path = os.path.join(logs_dir, "run.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source,
                "captured_at": stamp,
                "backend": backend_name,
                "mode": mode_name,
                "objects": int(args.objects),
                "n_pos": int(args.n_pos),
                "n_neg": int(args.n_neg),
                "multimask": bool(args.multimask),
                "frame": frame_path,
                "out_root": out_root,
                "masks": mask_paths,
                "modes": modes,
            },
            f,
            indent=2,
        )

    return out_root, frame_path, mask_paths, log_path


def main():
    parser = argparse.ArgumentParser(description="SAM2 live RTSP click/box prompt -> mask")
    parser.add_argument("--source", default="rtsp://192.168.5.201:8554/cam1", help="RTSP URL or video source")
    parser.add_argument("--backend", default=os.environ.get("SAM2_BACKEND", "sam2_repo"))
    parser.add_argument("--objects", type=int, default=1, help="Number of objects to segment per action")
    parser.add_argument("--n-pos", type=int, default=1)
    parser.add_argument("--n-neg", type=int, default=0)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Root output dir (default: sam-3d-objects/outputs)",
    )
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
    parser.add_argument(
        "--keep-buffered",
        action="store_true",
        help="Keep oldest queued frame instead of always replacing with newest",
    )
    parser.add_argument(
        "--ffmpeg-capture-options",
        default=None,
        help="Override OPENCV_FFMPEG_CAPTURE_OPTIONS string",
    )
    args = parser.parse_args()

    ffmpeg_opts = args.ffmpeg_capture_options
    if ffmpeg_opts is None:
        ffmpeg_opts = (
            "fflags;nobuffer|flags;low_delay|framedrop;1|probesize;32|analyzeduration;0|"
            f"rtbufsize;4M|rtsp_transport;{args.rtsp_transport}|stimeout;5000000|max_delay;500000|loglevel;quiet"
        )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts

    import cv2
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend_name, backend_obj = setup_sam2_backend(args.backend, device)

    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def capture_loop():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            if frame_queue.full() and not args.keep_buffered:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            if frame_queue.full() and args.keep_buffered:
                continue
            frame_queue.put(frame)

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    win = "SAM2 RTSP Live | p=points b=box m=mixed q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"[INFO] Live source: {args.source}")
    print(f"[INFO] Backend: {backend_name} on {device}")
    print("[INFO] Hotkeys: p=points, b=box, m=mixed, q=quit")

    latest = None
    try:
        while True:
            try:
                latest = frame_queue.get(timeout=0.02)
            except queue.Empty:
                pass

            if latest is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            display = latest.copy()
            cv2.putText(
                display,
                "p=points  b=box  m=mixed  q=quit",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            mode = None
            if key == ord("p"):
                mode = "points"
            elif key == ord("b"):
                mode = "box"
            elif key == ord("m"):
                mode = "mixed"

            if mode is None:
                continue

            frame_for_seg = latest.copy()
            image_rgb = frame_for_seg[:, :, ::-1].copy()

            print(f"[INFO] Running SAM2 ({mode})...")
            try:
                masks_out, modes = _predict_masks(
                    image_rgb=image_rgb,
                    backend=backend_name,
                    backend_obj=backend_obj,
                    mode=mode,
                    objects=max(1, int(args.objects)),
                    n_pos=int(args.n_pos),
                    n_neg=int(args.n_neg),
                    multimask=bool(args.multimask),
                )

                out_root, frame_path, mask_paths, log_path = _save_run(
                    frame_bgr=frame_for_seg,
                    source=args.source,
                    masks_out=masks_out,
                    modes=modes,
                    backend_name=backend_name,
                    mode_name=mode,
                    args=args,
                )

                print(f"[INFO] Saved frame: {frame_path}")
                for p in mask_paths:
                    print(f"[INFO] Saved mask: {p}")
                print(f"[INFO] Saved log: {log_path}")
                print(f"[INFO] Run root: {out_root}")
            except Exception as e:
                print(f"[ERROR] Segmentation failed: {e}")

    finally:
        stop_event.set()
        t.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
