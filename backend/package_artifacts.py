"""Create the small, local inference bundle used by the backend image.

Run from the repository root after the phase 1-4 rebuild has finished. This
copies trained model files and fixed held-out examples, not the SWAN-SF archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
BACKEND = Path(__file__).resolve().parent
MODELS = BACKEND / "models"
DEST = BACKEND / "data"
os.environ["HELION_DATA_DIR"] = str(SOURCE)
os.environ["HELION_SOURCE_ARTIFACTS"] = "1"

from backend import server  # noqa: E402


def copy(relative: Path) -> None:
    source = SOURCE / relative
    target = DEST / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    if server.DATA != SOURCE.resolve():
        raise RuntimeError("Package from the repository's original data directory.")
    models = server.catalog()
    if len(models) != 33:
        raise RuntimeError(f"Expected 33 verified runs; found {len(models)}.")
    conversion = json.loads((MODELS / "conversion_report.json").read_text(encoding="utf-8"))
    if len(conversion) != len(models) or {entry["model_id"] for entry in conversion} != set(models):
        raise RuntimeError("ONNX conversion report does not cover all saved runs.")
    for entry in conversion:
        model_file = MODELS / Path(entry["onnx_file"]).name
        original = models[entry["model_id"]]["_run_dir"] / models[entry["model_id"]]["_config"]["model_file"]
        if (not entry["verified"]
                or hashlib.sha256(model_file.read_bytes()).hexdigest() != entry["onnx_sha256"]
                or hashlib.sha256(original.read_bytes()).hexdigest() != entry["source_model_sha256"]):
            raise RuntimeError(f"ONNX conversion is missing or stale: {entry['model_id']}")
    if DEST.exists():
        if DEST.resolve() != (BACKEND / "data").resolve() or not DEST.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("Refusing to clean an unexpected bundle directory.")
        shutil.rmtree(DEST)
    copy(Path("leakage_summary.json"))
    copy(Path("phase2/horizon_targets_report.json"))
    report_target = DEST / "rebuild" / "onnx" / "conversion_report.json"
    report_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODELS / "conversion_report.json", report_target)
    for entry in conversion:
        target = DEST / entry["onnx_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODELS / Path(entry["onnx_file"]).name, target)
    for name in ("phase4_results.csv", "phase4_results.json"):
        source = ROOT / "reports" / "phase4" / name
        target = DEST / "reports" / "phase4" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for split in ("region_split", "time_split"):
        for kind in ("baseline", "temporal"):
            copy(Path("phase1") / f"normalization_{split}_{kind}.json")
    example_count = 0
    for item in models.values():
        run = item["_run_dir"].relative_to(SOURCE)
        for name in ("config.json", "status.json", "metrics.json"):
            copy(run / name)
        if item["diagnostic_only"]:
            continue
        for kind in ("flare", "quiet"):
            sample = server.example(item["id"], kind)
            target = DEST / "examples" / f"{item['id'].replace(':', '_')}_{kind}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(sample, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            example_count += 1
    manifest = {"model_count": len(models), "example_count": example_count,
                "files": {str(path.relative_to(DEST)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(DEST.rglob("*")) if path.is_file() and path.name != "manifest.json"}}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = BACKEND / "assets" / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DEST))
    print(f"Packaged {len(models)} verified ONNX models and {example_count} held-out examples into {bundle}")


if __name__ == "__main__":
    main()
