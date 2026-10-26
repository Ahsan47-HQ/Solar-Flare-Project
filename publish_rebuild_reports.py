"""Publish verified rebuilt results over the obsolete Phase 2--4 placeholders."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import rebuild_common as c


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for label, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [head, rule]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row[key]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def publish() -> None:
    source = c.REPORTS / "phase4" / "verified_results.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["warnings"]:
        raise ValueError("Cannot publish while saved runs fail verification")
    rows = payload["results"]
    p2 = [r for r in rows if r["phase"] == "phase2"]
    p3 = [r for r in rows if r["phase"] == "phase3"]
    phase2 = c.ROOT / "reports" / "phase2"
    phase3 = c.ROOT / "reports" / "phase3"
    phase4 = c.ROOT / "reports" / "phase4"
    shutil.copyfile(c.REPORTS / "phase2" / "verified_results.csv", phase2 / "classical_model_metrics.csv")
    shutil.copyfile(c.REPORTS / "phase3" / "verified_results.csv", phase3 / "phase3_metrics.csv")
    shutil.copyfile(c.REPORTS / "phase4" / "verified_results.csv", phase4 / "phase4_results.csv")
    shutil.copyfile(source, phase4 / "phase4_results.json")
    shutil.copyfile(c.REPORTS / "phase4" / "verified_summary.md", phase4 / "phase4_summary.md")

    cols2 = [("Split", "split"), ("Features", "features"), ("Model", "model"),
             ("Test N", "n_test"), ("FL", "positive"), ("AP", "ap"),
             ("Recall", "recall"), ("Precision", "precision"), ("TSS", "tss"), ("HSS", "hss")]
    write(phase2 / "classical_model_results.md", "\n".join([
        "# Phase 2: verified full-data model results", "",
        "M/X flare peak in the next 24 hours; 12-hour magnetic observation windows. "
        "The random split is a leakage diagnostic only. The decision threshold was selected on validation F1.", "",
        table(p2, cols2), "",
        f"All {len(p2)} runs passed model reload, data-hash, held-out ID, and metric checks. "
        "Detailed configurations and probabilities are under `data/rebuild/runs/phase2/`.",
        "The random-split AP must not be treated as independent-region generalization.", "",
    ]))
    write(phase2 / "model_reload_verification.md", "\n".join([
        "# Phase 2 model reload verification", "",
        f"All {len(p2)} Phase 2 models passed saved artifact, current Phase 1 input hash, "
        "held-out split identity, metric recomputation, and 64-row model reload prediction checks.",
        "See `reports/rebuild/phase4/verified_results.json` for run directories and scores.", "",
    ]))
    inventory = []
    for row in p2:
        run_dir = Path(row["run_dir"])
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = run_dir / config["model_file"]
        inventory.append({"run_dir": str(run_dir), "model_file": model.name,
                          "model_sha256": c.file_hash(model), "reload_status": "verified"})
    write(phase2 / "artifact_inventory.json", json.dumps(inventory, indent=2))
    write(phase2 / "experiment_manifest.json", json.dumps({
        "phase": "phase2", "source": "verified rebuild", "runs": p2,
        "label_definition": payload["label_definition"],
    }, indent=2))
    write(phase2 / "requirements_audit.md", "\n".join([
        "# Phase 2 current audit", "",
        "The seven archives and Phase 1 arrays are present locally. Full-data Logistic Regression, "
        "Random Forest, XGBoost, and ANN runs are saved and independently reloaded on held-out rows.",
        "XGBoost uses bounded CUDA histogram training and validation early stopping. "
        "The ANN uses CPU because native Windows TensorFlow 2.21 has no CUDA GPU support.",
        "A separate validation-only XGBoost depth and ANN width sensitivity study is in "
        "`validation_sensitivity.json`; it does not alter held-out scores.",
        f"{len(p2)} verified Phase 2 configurations are in `classical_model_metrics.csv`.", "",
    ]))

    cols3 = [("Lead", "lead_hours"), ("Split", "split"), ("Features", "features"),
             ("Test N", "n_test"), ("FL", "positive"), ("FL regions", "positive_regions"),
             ("AP", "ap"), ("Recall", "recall"), ("TSS", "tss"), ("HSS", "hss")]
    for name, leads in (("short_horizon_results.md", (6, 12)),
                        ("long_horizon_results.md", (24, 48))):
        subset = [r for r in p3 if r["lead_hours"] in leads]
        write(phase3 / name, "\n".join([
            f"# Phase 3: verified {'short' if 6 in leads else 'long'} lead-time results", "",
            "A positive label is an M/X flare peak in `(issue+H, issue+H+24h]`. "
            "All four H values use the same eligible windows. XGBoost was trained with bounded CUDA resources.", "",
            table(sorted(subset, key=lambda r: (r["lead_hours"], r["split"], r["features"])), cols3), "",
            "The target is peak time, so this does not directly measure lead to flare onset. "
            "The 48-hour cohort excludes windows without complete future coverage.", "",
        ]))
    comparisons = payload["paired_comparisons"]
    comparison_rows = [{"lead": f"{x['lead_hours']}h", "split": x["split"],
                        "ap_difference": x["difference_ap"],
                        "ci": f"[{x['ci95_region_bootstrap'][0]:.3f}, {x['ci95_region_bootstrap'][1]:.3f}]"}
                       for x in comparisons]
    write(phase3 / "feature_comparison_summary.md", "\n".join([
        "# Phase 3: baseline versus temporal features", "",
        "Paired XGBoost AP differences on identical held-out windows; temporal minus baseline. "
        "500 bootstrap replicates resample active regions, preserving window dependence.", "",
        table(comparison_rows, [("Lead", "lead"), ("Split", "split"),
                                ("AP difference", "ap_difference"), ("95% region CI", "ci")]), "",
        "Every interval includes zero. These runs do not establish a reliable AP benefit from "
        "adding the 120 temporal features. See `reports/rebuild/phase3/` for AP/TSS plots.", "",
    ]))
    write(phase4 / "phase4_run_manifest.json", json.dumps({
        "phase": "phase4", "source": "verified rebuild", "verified_run_count": len(rows),
        "rejected_run_count": len(payload["warnings"]),
        "run_directories": [r["run_dir"] for r in rows],
        "label_definition": payload["label_definition"],
    }, indent=2))
    print(f"Published {len(p2)} Phase 2 and {len(p3)} Phase 3 verified runs")


if __name__ == "__main__":
    publish()
