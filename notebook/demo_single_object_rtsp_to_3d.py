#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn RTSP SAM2 capture runs into SAM-3D single-object results.

Flow:
1) choose a saved run/mask (GUI by default)
2) run SAM-3D inference and save PLY
3) optionally render a GIF
4) auto-open a 3D viewer for the generated splat
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PREFERRED_PYTHON = Path("/home/mnc/micromamba/envs/sam3d-objects/bin/python")

# Avoid importing local source trees from the workspace root (e.g. cumm-src/)
# when this script is launched from /home/mnc/mccv. The notebook doesn't have
# this issue because its kernel cwd/path differs.
_cwd = Path.cwd().resolve()
for _bad in ("", str(_cwd)):
    while _bad in sys.path:
        sys.path.remove(_bad)

# Ensure subprocess tools spawned by cumm/spconv (e.g., ninja) are resolvable
# even when VSCode launches this script with a direct interpreter path.
_env_bin = str(Path(sys.executable).resolve().parent)
_cur_path = os.environ.get("PATH", "")
if not _cur_path.startswith(_env_bin + os.pathsep):
    os.environ["PATH"] = _env_bin + os.pathsep + _cur_path

# Mirror notebook behavior for CUDA discovery before importing `inference.py`.
def _pick_cuda_home() -> Optional[Path]:
    candidates = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(Path(os.environ["CUDA_HOME"]))
    candidates.extend(
        [
            Path("/usr/local/cuda-12.8"),
            Path("/usr/local/cuda"),
        ]
    )
    for c in candidates:
        if (c / "include" / "curand.h").exists():
            return c
    return None


_cuda_home = _pick_cuda_home()
if _cuda_home is not None:
    os.environ["CUDA_HOME"] = str(_cuda_home)
    cuda_bin = str(_cuda_home / "bin")
    cuda_lib = str(_cuda_home / "lib64")
    if not os.environ["PATH"].startswith(cuda_bin + os.pathsep):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    os.environ["LD_LIBRARY_PATH"] = (
        cuda_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ.setdefault("FORCE_CUDA", "1")

# Match notebook-style environment setup.
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(REPO_ROOT / ".torch_extensions"))
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _natural_key(p: Path):
    stem = p.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "scene"


def _import_inference_api():
    try:
        from inference import (
            Inference,
            debug_inference_and_save,
            load_image,
            load_mask,
            make_scene,
            ready_gaussian_for_video_rendering,
            render_video,
            interactive_visualizer,
        )
        return (
            Inference,
            debug_inference_and_save,
            load_image,
            load_mask,
            make_scene,
            ready_gaussian_for_video_rendering,
            render_video,
            interactive_visualizer,
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        if "utils3d" in missing:
            # VSCode sometimes runs with a different interpreter than the selected notebook kernel.
            # If the known-good env exists, transparently relaunch once with that interpreter.
            if (
                os.environ.get("SAM3D_REEXECED") != "1"
                and PREFERRED_PYTHON.exists()
                and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve()
            ):
                print(
                    "[INFO] 'utils3d' missing in current interpreter; relaunching with "
                    f"{PREFERRED_PYTHON}"
                )
                os.environ["SAM3D_REEXECED"] = "1"
                os.execv(str(PREFERRED_PYTHON), [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
            raise RuntimeError(
                "Missing dependency 'utils3d' for this interpreter.\n"
                f"Current python: {sys.executable}\n"
                "Use your SAM3D micromamba env interpreter instead, e.g.:\n"
                "  /home/mnc/micromamba/envs/sam3d-objects/bin/python "
                "sam-3d-objects/notebook/demo_single_object_rtsp_to_3d.py"
            ) from e
        raise


def discover_runs(outputs_root: Path, include_legacy: bool = False) -> List[Dict]:
    logs = sorted(
        outputs_root.glob("**/logs/run.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    runs: List[Dict] = []
    for log_path in logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        out_root = Path(data.get("out_root", log_path.parents[1]))
        if not out_root.is_absolute():
            out_root = (outputs_root / out_root).resolve()

        frame = data.get("frame")
        frame_path = Path(frame) if frame else (out_root / "frame.png")
        if not frame_path.is_absolute():
            frame_path = (out_root / frame_path).resolve()

        is_rtsp_style = frame_path.exists()
        if not include_legacy and not is_rtsp_style:
            continue

        if not frame_path.exists() and data.get("image_path"):
            maybe_img = Path(data["image_path"])
            if maybe_img.exists():
                frame_path = maybe_img

        if not frame_path.exists():
            continue

        mask_paths: List[Path] = []
        masks_field = data.get("masks", [])
        if isinstance(masks_field, list) and masks_field:
            for m in masks_field:
                mp = Path(m)
                if not mp.is_absolute():
                    mp = (out_root / mp).resolve()
                if mp.exists():
                    mask_paths.append(mp)

        if not mask_paths:
            mask_dir = out_root / "masks"
            if mask_dir.exists():
                mask_paths = sorted(mask_dir.glob("*.png"), key=_natural_key)

        if not mask_paths:
            continue

        runs.append(
            {
                "log": log_path,
                "out_root": out_root,
                "frame": frame_path,
                "masks": mask_paths,
                "captured_at": data.get("captured_at", ""),
                "mode": data.get("mode", ""),
                "backend": data.get("backend", ""),
                "source": data.get("source", data.get("image_path", "")),
                "raw": data,
            }
        )

    return runs


def print_runs(runs: List[Dict], limit: int):
    print("\nAvailable runs:\n")
    for idx, run in enumerate(runs[:limit]):
        out_name = run["out_root"].name
        parent_name = run["out_root"].parent.name
        print(
            f"[{idx}] {parent_name}/{out_name} | masks={len(run['masks'])} "
            f"| mode={run['mode'] or '-'} | time={run['captured_at'] or '-'}"
        )
    print()


def choose_run_mask_gui(
    runs: List[Dict],
    default_run: int = 0,
    default_mask: int = 0,
) -> Tuple[int, int]:
    try:
        import tkinter as tk
    except Exception as e:
        raise RuntimeError("tkinter is not available for GUI selection.") from e

    default_run = max(0, min(default_run, len(runs) - 1))
    state = {"run": default_run, "mask": default_mask, "ok": False}

    root = tk.Tk()
    root.title("Select RTSP Capture and Mask")
    root.geometry("1100x650")

    top = tk.Frame(root)
    top.pack(fill="both", expand=True, padx=10, pady=10)

    left = tk.Frame(top)
    left.pack(side="left", fill="both", expand=True, padx=(0, 8))
    right = tk.Frame(top)
    right.pack(side="left", fill="both", expand=True, padx=(8, 0))

    tk.Label(left, text="Runs").pack(anchor="w")
    run_list = tk.Listbox(left, exportselection=False)
    run_list.pack(fill="both", expand=True)

    for i, run in enumerate(runs):
        out_name = run["out_root"].name
        parent_name = run["out_root"].parent.name
        run_list.insert(
            tk.END,
            f"[{i}] {parent_name}/{out_name} | masks={len(run['masks'])} | "
            f"mode={run['mode'] or '-'} | time={run['captured_at'] or '-'}",
        )

    tk.Label(right, text="Masks").pack(anchor="w")
    mask_list = tk.Listbox(right, exportselection=False)
    mask_list.pack(fill="both", expand=True)

    details_var = tk.StringVar(value="")
    details = tk.Label(root, textvariable=details_var, justify="left", anchor="w")
    details.pack(fill="x", padx=10, pady=(0, 8))

    def _refresh_masks():
        run_idx = state["run"]
        masks = runs[run_idx]["masks"]
        mask_list.delete(0, tk.END)
        for i, p in enumerate(masks):
            mask_list.insert(tk.END, f"[{i}] {p.name}")
        if masks:
            pick = max(0, min(state["mask"], len(masks) - 1))
            mask_list.selection_set(pick)
            state["mask"] = pick
        _refresh_details()

    def _refresh_details():
        run = runs[state["run"]]
        frame = run["frame"]
        masks = run["masks"]
        mask_name = masks[state["mask"]].name if masks else "-"
        details_var.set(
            f"Run: {run['out_root']}\n"
            f"Frame: {frame}\n"
            f"Mask: {mask_name}"
        )

    def _on_run_select(event=None):
        sel = run_list.curselection()
        if not sel:
            return
        state["run"] = int(sel[0])
        state["mask"] = 0
        _refresh_masks()

    def _on_mask_select(event=None):
        sel = mask_list.curselection()
        if not sel:
            return
        state["mask"] = int(sel[0])
        _refresh_details()

    def _accept(event=None):
        state["ok"] = True
        root.destroy()

    def _cancel(event=None):
        state["ok"] = False
        root.destroy()

    run_list.bind("<<ListboxSelect>>", _on_run_select)
    mask_list.bind("<<ListboxSelect>>", _on_mask_select)

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=10, pady=(0, 10))
    tk.Button(btns, text="Run", command=_accept).pack(side="left")
    tk.Button(btns, text="Cancel", command=_cancel).pack(side="left", padx=(8, 0))

    run_list.selection_set(default_run)
    _refresh_masks()
    root.bind("<Return>", _accept)
    root.bind("<Escape>", _cancel)
    root.mainloop()

    if not state["ok"]:
        raise RuntimeError("Selection cancelled.")
    return int(state["run"]), int(state["mask"])


def _choose_index(prompt: str, maximum: int, default: int = 0) -> int:
    while True:
        raw = input(f"{prompt} [0-{maximum}] (default {default}): ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= val <= maximum:
            return val
        print("Out of range.")


def main():
    parser = argparse.ArgumentParser(description="Run SAM-3D from RTSP SAM2 saved runs")
    parser.add_argument(
        "--outputs-root",
        default=str(REPO_ROOT / "outputs"),
        help="Folder containing saved runs (default: sam-3d-objects/outputs)",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "checkpoints" / "hf" / "pipeline.yaml"),
        help="SAM-3D pipeline config path",
    )
    parser.add_argument(
        "--gaussian-out",
        default=str(SCRIPT_DIR / "gaussians" / "single"),
        help="Output folder for .ply and .gif",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-inference-steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Compile inference model")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF rendering")
    parser.add_argument("--gif-resolution", type=int, default=512)
    parser.add_argument("--gif-frames", type=int, default=180)
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--gif-radius", type=float, default=1.0)
    parser.add_argument("--gif-fov", type=float, default=60.0)
    parser.add_argument("--gif-pitch", type=float, default=15.0)
    parser.add_argument("--gif-yaw-start", type=float, default=-45.0)
    parser.add_argument("--list-limit", type=int, default=30)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--mask-index", type=int, default=None)
    parser.add_argument("--cli-select", action="store_true", help="Use terminal prompts instead of GUI")
    parser.add_argument("--list-only", action="store_true", help="List runs and exit")
    parser.add_argument("--no-visualizer", action="store_true", help="Do not auto-open 3D viewer")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also include older non-RTSP runs if present",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    if not outputs_root.exists():
        raise FileNotFoundError(f"Outputs root not found: {outputs_root}")

    runs = discover_runs(outputs_root, include_legacy=args.include_legacy)
    if not runs:
        raise RuntimeError(
            f"No valid runs found in {outputs_root}. "
            "Run demo_single_object_sam2_rtsp_click.py first."
        )

    run_index = args.run_index
    mask_index = args.mask_index
    if args.list_only:
        print_runs(runs, args.list_limit)
        return

    if (run_index is None or mask_index is None) and not args.cli_select:
        try:
            run_index, mask_index = choose_run_mask_gui(
                runs,
                default_run=0 if run_index is None else int(run_index),
                default_mask=0 if mask_index is None else int(mask_index),
            )
        except Exception as e:
            print(f"[WARN] GUI selection unavailable ({e}). Falling back to terminal prompts.")

    if run_index is None:
        print_runs(runs, args.list_limit)
        run_index = _choose_index("Choose run index", min(args.list_limit, len(runs)) - 1, default=0)

    if not (0 <= run_index < len(runs)):
        raise ValueError(f"run-index out of range: {run_index}")

    selected = runs[run_index]
    frame_path: Path = selected["frame"]
    mask_paths: List[Path] = selected["masks"]

    print(f"\nSelected run: {selected['out_root']}")
    print(f"Frame: {frame_path}")
    print(f"Masks: {len(mask_paths)}")
    for i, mp in enumerate(mask_paths):
        print(f"  [{i}] {mp.name}")

    if mask_index is None:
        mask_index = _choose_index("Choose mask index", len(mask_paths) - 1, default=0)

    if not (0 <= mask_index < len(mask_paths)):
        raise ValueError(f"mask-index out of range: {mask_index}")

    mask_path = mask_paths[mask_index]

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    gaussian_out = Path(args.gaussian_out).expanduser().resolve()
    gaussian_out.mkdir(parents=True, exist_ok=True)

    (
        Inference,
        debug_inference_and_save,
        load_image,
        load_mask,
        make_scene,
        ready_gaussian_for_video_rendering,
        render_video,
        interactive_visualizer,
    ) = _import_inference_api()

    print(f"\n[INFO] Loading SAM-3D pipeline from {config_path}")
    inference = Inference(str(config_path), compile=args.compile)

    print("[INFO] Loading selected frame and mask")
    image = load_image(str(frame_path))
    mask = load_mask(str(mask_path))

    scene_label = _safe_name(f"{selected['out_root'].parent.name}_{selected['out_root'].name}_m{mask_index}")

    print("[INFO] Running inference")
    output = debug_inference_and_save(
        inference_fn=lambda img, msk, seed=42: inference(
            img,
            msk,
            seed=seed,
            stage1_inference_steps=args.stage1_inference_steps,
        ),
        image=image,
        mask=mask,
        out_dir=str(gaussian_out),
        image_name=scene_label,
        seed=args.seed,
    )

    result = {
        "run_root": str(selected["out_root"]),
        "frame": str(frame_path),
        "mask": str(mask_path),
        "mask_index": int(mask_index),
        "seed": int(args.seed),
        "ply": str(gaussian_out / f"{scene_label}.ply"),
    }

    if not args.no_gif:
        try:
            import imageio
        except Exception as e:
            raise RuntimeError(
                "GIF rendering requested but imageio is not installed. "
                "Install it or pass --no-gif."
            ) from e
        print("[INFO] Rendering GIF")
        scene_gs = make_scene(output)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)
        video = render_video(
            scene_gs,
            r=args.gif_radius,
            fov=args.gif_fov,
            pitch_deg=args.gif_pitch,
            yaw_start_deg=args.gif_yaw_start,
            resolution=args.gif_resolution,
            num_frames=args.gif_frames,
        )["color"]

        gif_path = gaussian_out / f"{scene_label}.gif"
        imageio.mimsave(
            gif_path,
            video,
            format="GIF",
            duration=1000 / max(1, args.gif_fps),
            loop=0,
        )
        result["gif"] = str(gif_path)
        print(f"[INFO] GIF saved: {gif_path}")

    meta_path = gaussian_out / f"{scene_label}.json"
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"PLY: {result['ply']}")
    if "gif" in result:
        print(f"GIF: {result['gif']}")
    print(f"Metadata: {meta_path}")

    if not args.no_visualizer:
        print("[INFO] Launching 3D visualizer...")
        interactive_visualizer(result["ply"])


if __name__ == "__main__":
    main()
