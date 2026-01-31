#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 (click/box prompt) as a plain Python script.
Single- or multi-object mask creation with click or box prompt.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image


REQUIRE_CV2 = True  # Force OpenCV UI for interactive prompts (no matplotlib fallback)


def load_image_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_mask(mask_bool, image_path, out_path=None):
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(image_path), "0.png")
    Image.fromarray(mask_uint8).save(out_path)
    return out_path


def overlay_mask(image_rgb, mask_bool, alpha=0.5):
    img = image_rgb.astype(np.float32).copy()
    overlay = img.copy()
    overlay[mask_bool] = np.array([255, 0, 0], dtype=np.float32)
    out = (img * (1.0 - alpha) + overlay * alpha).astype(np.uint8)
    return out


def build_output_dirs(image_path, root_dir=None):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(__file__), "outputs")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(root_dir, image_name, run_id)
    masks_dir = os.path.join(out_root, "masks")
    overlays_dir = os.path.join(out_root, "overlays")
    logs_dir = os.path.join(out_root, "logs")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return out_root, masks_dir, overlays_dir, logs_dir


def get_points_cv2(img_rgb, n_pos, n_neg):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("cv2 not available. Install opencv-python.") from e

    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[..., None], 3, axis=2)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[..., :3]
    img_rgb = np.ascontiguousarray(img_rgb)
    img_bgr = img_rgb[..., ::-1].copy()

    points = []
    labels = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN and (n_neg is None or n_neg > 0):
            points.append((x, y))
            labels.append(0)

    win = "Click: left=positive, right=negative (press q to finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = np.ascontiguousarray(img_bgr.copy())
        for (x, y), lab in zip(points, labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(disp, (x, y), 4, color, -1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyWindow(win)
    return points, labels


def get_points_mpl(img_rgb):
    import matplotlib.pyplot as plt

    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[..., None], 3, axis=2)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[..., :3]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click: left=positive, right=negative; close window when done")
    points = []
    labels = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            points.append((int(event.xdata), int(event.ydata)))
            labels.append(1)
            ax.plot(event.xdata, event.ydata, "go")
        elif event.button == 3:
            points.append((int(event.xdata), int(event.ydata)))
            labels.append(0)
            ax.plot(event.xdata, event.ydata, "ro")
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return points, labels


def get_points(img_rgb, n_pos, n_neg, force_mpl=False):
    if force_mpl:
        return get_points_mpl(img_rgb)
    try:
        return get_points_cv2(img_rgb, n_pos, n_neg)
    except Exception as e:
        if REQUIRE_CV2:
            raise RuntimeError(
                "OpenCV UI is required but cv2 is not available. "
                "Install opencv-python and rerun."
            ) from e
        return get_points_mpl(img_rgb)


def get_box_cv2(img_rgb):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("cv2 not available. Install opencv-python.") from e

    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[..., None], 3, axis=2)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[..., :3]
    img_rgb = np.ascontiguousarray(img_rgb)
    img_bgr = img_rgb[..., ::-1].copy()

    win = "Draw box: click-drag (press q to finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    box = None
    drawing = False
    start = (0, 0)

    def on_mouse(event, x, y, flags, param):
        nonlocal box, drawing, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            box = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x0, y0 = start
            box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = start
            box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = np.ascontiguousarray(img_bgr.copy())
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyWindow(win)
    return box


def get_box_mpl(img_rgb):
    import matplotlib.pyplot as plt

    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[..., None], 3, axis=2)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[..., :3]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Box: click two corners; close window when done")
    pts = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            pts.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, "go")
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    if len(pts) < 2:
        return None
    (x0, y0), (x1, y1) = pts[:2]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def get_box(img_rgb, force_mpl=False):
    if force_mpl:
        return get_box_mpl(img_rgb)
    try:
        return get_box_cv2(img_rgb)
    except Exception as e:
        if REQUIRE_CV2:
            raise RuntimeError(
                "OpenCV UI is required but cv2 is not available. "
                "Install opencv-python and rerun."
            ) from e
        return get_box_mpl(img_rgb)


def get_prompts_cv2_mixed(img_rgb, objects, n_pos, n_neg):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("cv2 not available. Install opencv-python.") from e

    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[..., None], 3, axis=2)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[..., :3]
    img_rgb = np.ascontiguousarray(img_rgb)
    img_bgr = img_rgb[..., ::-1].copy()

    h, w = img_bgr.shape[:2]
    btn_w, btn_h = 140, 32
    margin = 10
    points_btn = (margin, margin, margin + btn_w, margin + btn_h)
    box_btn = (margin + btn_w + 10, margin, margin + 2 * btn_w + 10, margin + btn_h)

    def in_rect(x, y, rect):
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def draw_button(img, rect, label, active=False):
        x0, y0, x1, y1 = rect
        color = (0, 200, 0) if active else (60, 60, 60)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(
            img,
            label,
            (x0 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    all_point_coords = []
    all_point_labels = []
    all_boxes = []
    all_modes = []

    win = "SAM2 mixed prompt: click buttons to switch, press n=next, q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for obj_idx in range(objects):
        mode = "points"
        points = []
        labels = []
        box = None
        drawing = False
        start = (0, 0)

        def on_mouse(event, x, y, flags, param):
            nonlocal mode, points, labels, box, drawing, start
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_rect(x, y, points_btn):
                    mode = "points"
                    return
                if in_rect(x, y, box_btn):
                    mode = "box"
                    return
                if mode == "points":
                    points.append((x, y))
                    labels.append(1)
                else:
                    drawing = True
                    start = (x, y)
                    box = None
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                x0, y0 = start
                box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                x0, y0 = start
                box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            elif event == cv2.EVENT_RBUTTONDOWN and mode == "points":
                if n_neg is None or n_neg > 0:
                    points.append((x, y))
                    labels.append(0)

        cv2.setMouseCallback(win, on_mouse)

        while True:
            disp = np.ascontiguousarray(img_bgr.copy())
            draw_button(disp, points_btn, "POINTS", active=(mode == "points"))
            draw_button(disp, box_btn, "BOX", active=(mode == "box"))
            cv2.putText(
                disp,
                f"Object {obj_idx+1}/{objects} | mode: {mode} | n=next, q=quit",
                (margin, margin + btn_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for (x, y), lab in zip(points, labels):
                color = (0, 255, 0) if lab == 1 else (0, 0, 255)
                cv2.circle(disp, (x, y), 4, color, -1)
            if box is not None:
                x0, y0, x1, y1 = box
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                raise RuntimeError("User cancelled.")
            if key == ord("n"):
                if mode == "points" and len(points) == 0:
                    print(f"No points selected for object {obj_idx+1}.")
                    continue
                if mode == "box" and box is None:
                    print(f"No box selected for object {obj_idx+1}.")
                    continue
                break

        all_modes.append(mode)
        if mode == "points":
            all_point_coords.append(np.array(points, dtype=np.float32))
            all_point_labels.append(np.array(labels, dtype=np.int32))
            all_boxes.append(None)
        else:
            all_point_coords.append(None)
            all_point_labels.append(None)
            all_boxes.append(np.array(box, dtype=np.float32))

    cv2.destroyWindow(win)
    return all_point_coords, all_point_labels, all_boxes, all_modes


def setup_sam2_backend(backend, device):
    if backend == "sam2_repo":
        sam2_root = os.environ.get("SAM2_ROOT", "/home/mnc/mccv/sam2")
        sam2_ckpt = os.environ.get(
            "SAM2_CHECKPOINT",
            os.path.join(sam2_root, "checkpoints", "sam2.1_hiera_large.pt"),
        )
        if os.path.basename(sam2_ckpt).startswith("sam2.1_"):
            sam2_cfg = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
        else:
            sam2_cfg = os.environ.get("SAM2_CFG", "configs/sam2/sam2_hiera_l.yaml")

        if os.path.isabs(sam2_cfg) and "configs/" in sam2_cfg:
            sam2_cfg = "configs/" + sam2_cfg.split("configs/")[-1]
        if not sam2_cfg.startswith("configs/"):
            sam2_cfg = "configs/" + sam2_cfg

        if not os.path.exists(sam2_ckpt):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt}")

        sys.path.append(sam2_root)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_predictor.model.eval()
        return ("sam2_repo", sam2_predictor)

    if backend == "transformers":
        model_id = os.environ.get("SAM2_MODEL_ID", "facebook/sam2-hiera-large")
        cache_dir = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
        hf_token = os.environ.get("HF_TOKEN", None)
        try:
            from transformers import Sam2Processor, Sam2Model
        except Exception as e:
            raise ImportError(
                "Sam2Processor/Sam2Model not available. Use SAM2_BACKEND='sam2_repo'."
            ) from e

        import torch

        model = Sam2Model.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=hf_token,
            torch_dtype=(torch.float16 if device == "cuda" else None),
        ).to(device).eval()
        processor = Sam2Processor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
        return ("transformers", (model, processor))

    raise ValueError("backend must be 'sam2_repo' or 'transformers'")


def main():
    parser = argparse.ArgumentParser(description="SAM2 click prompt -> single-object mask")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--backend", default=os.environ.get("SAM2_BACKEND", "sam2_repo"))
    parser.add_argument("--mode", choices=["points", "box", "mixed"], default="mixed")
    parser.add_argument("--n-pos", type=int, default=1)
    parser.add_argument("--n-neg", type=int, default=0)
    parser.add_argument("--objects", type=int, default=1, help="Number of objects to segment")
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--out-mask", default=None, help="Output mask path (PNG) or dir for multi")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Root output dir (default: sam-3d-objects/outputs)",
    )
    parser.add_argument("--show", action="store_true", help="Show image + mask preview")
    parser.add_argument("--mpl", action="store_true", help="Force matplotlib click UI")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        default_image = "/home/mnc/Downloads/IMG_3792.jpg"
        if os.path.exists(default_image):
            image_path = default_image
        else:
            raise FileNotFoundError(
                "No --image provided and default image not found. "
                "Pass --image /path/to/image."
            )
    image_rgb = load_image_rgb(image_path)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend, backend_obj = setup_sam2_backend(args.backend, device)

    objects = max(1, int(args.objects))
    all_point_coords = []
    all_point_labels = []
    all_boxes = []
    all_modes = []

    if args.mode == "mixed":
        if args.mpl:
            raise RuntimeError("Mixed mode requires OpenCV UI. Remove --mpl.")
        all_point_coords, all_point_labels, all_boxes, all_modes = get_prompts_cv2_mixed(
            image_rgb, objects, args.n_pos, args.n_neg
        )
    elif args.mode == "points":
        for i in range(objects):
            pts, labels = get_points(
                image_rgb, args.n_pos, args.n_neg, force_mpl=args.mpl
            )
            if len(pts) == 0:
                raise RuntimeError(f"No points selected for object {i+1}.")
            all_point_coords.append(np.array(pts, dtype=np.float32))
            all_point_labels.append(np.array(labels, dtype=np.int32))
            all_boxes.append(None)
            all_modes.append("points")
    else:
        for i in range(objects):
            box = get_box(image_rgb, force_mpl=args.mpl)
            if box is None:
                raise RuntimeError(f"No box selected for object {i+1}.")
            all_boxes.append(np.array(box, dtype=np.float32))
            all_point_coords.append(None)
            all_point_labels.append(None)
            all_modes.append("box")

    masks_out = []

    if backend == "sam2_repo":
        sam2_predictor = backend_obj
        sam2_predictor.set_image(np.array(image_rgb))
        for i in range(objects):
            masks, scores, logits = sam2_predictor.predict(
                point_coords=all_point_coords[i],
                point_labels=all_point_labels[i],
                box=all_boxes[i],
                multimask_output=args.multimask,
            )
            if scores is not None:
                mask_idx = int(np.argmax(scores))
            else:
                mask_idx = 0
            masks_out.append(masks[mask_idx].astype(bool))
    else:
        model, processor = backend_obj
        input_points = None
        input_labels = None
        input_boxes = None
        if args.mode == "points":
            input_points = [[pts.tolist() for pts in all_point_coords]]
            input_labels = [[labs.tolist() for labs in all_point_labels]]
        if args.mode == "box":
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
            outputs = model(**inputs, multimask_output=args.multimask)

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

    out_root, masks_dir, overlays_dir, logs_dir = build_output_dirs(
        image_path, root_dir=args.out_root
    )

    mask_paths = []
    if args.out_mask:
        if args.out_mask.lower().endswith(".png") and objects == 1:
            mask_paths = [save_mask(masks_out[0], image_path, out_path=args.out_mask)]
        elif args.out_mask.lower().endswith(".png") and objects > 1:
            base, ext = os.path.splitext(args.out_mask)
            for i, m in enumerate(masks_out):
                mask_paths.append(save_mask(m, image_path, out_path=f"{base}_{i}{ext}"))
        else:
            out_dir = args.out_mask
            os.makedirs(out_dir, exist_ok=True)
            for i, m in enumerate(masks_out):
                mask_paths.append(save_mask(m, image_path, out_path=os.path.join(out_dir, f"{i}.png")))
    else:
        for i, m in enumerate(masks_out):
            mask_paths.append(save_mask(m, image_path, out_path=os.path.join(masks_dir, f"{i}.png")))

    for p in mask_paths:
        print("Saved mask:", p)

    if args.show:
        import matplotlib.pyplot as plt

        n = len(masks_out)
        fig, ax = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
        ax[0].imshow(image_rgb)
        ax[0].set_title("Image")
        ax[0].axis("off")
        for i, m in enumerate(masks_out, start=1):
            overlay = overlay_mask(image_rgb, m)
            overlay_path = os.path.join(overlays_dir, f"{i-1}.png")
            Image.fromarray(overlay).save(overlay_path)
            ax[i].imshow(overlay)
            ax[i].set_title(f"Mask {i-1}")
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()

    log_path = os.path.join(logs_dir, "run.json")
    log_data = {
        "image_path": image_path,
        "backend": args.backend,
        "mode": args.mode,
        "objects": objects,
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "multimask": bool(args.multimask),
        "out_root": out_root,
        "masks": mask_paths,
        "modes": all_modes,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print("Saved log:", log_path)


if __name__ == "__main__":
    main()
