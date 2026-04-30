#!/usr/bin/env python3
"""Export a single machine-readable correction-metrics summary JSON.

"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_TEX = REPO_ROOT / "paper" / "main.tex"
OUTPUT_JSON = REPO_ROOT / "outputs" / "analysis" / "correction_metrics_final.json"
OUTPUTS_ROOT = Path(os.getenv("BAMF_ECO_OUTPUTS_ROOT", str(REPO_ROOT / "outputs")))
CORRECTION_ANALYSIS_JSON = OUTPUTS_ROOT / "bamf_eco_promotion" / "correction_analysis.json"
ABLATION_JSON = OUTPUTS_ROOT / "analysis" / "ablation_results.json"


def parse_reported_metrics_from_paper(tex: str) -> dict:
    out = {
        "spearman_rho": None,
        "kendall_tau": None,
        "p_spearman_lt": None,
        "p_kendall_lt": None,
        "r2_map": None,
        "matched_configurations": None,
        "top5_recall": None,
        "top5_hits": None,
        "top5_total": None,
    }

    m = re.search(r"Spearman \$\\rho\s*=\s*([0-9.]+)\$", tex)
    if m:
        out["spearman_rho"] = float(m.group(1))

    m = re.search(r"Kendall \$\\tau\s*=\s*([0-9.]+)\$", tex)
    if m:
        out["kendall_tau"] = float(m.group(1))

    m = re.search(r"Spearman \$\\rho\s*=\s*[0-9.]+\$\s*\(\$p\s*<\s*10\^{-([0-9]+)}\$\)", tex)
    if m:
        out["p_spearman_lt"] = f"1e-{m.group(1)}"

    m = re.search(r"Kendall \$\\tau\s*=\s*[0-9.]+\$\s*\(\$p\s*<\s*10\^{-([0-9]+)}\$\)", tex)
    if m:
        out["p_kendall_lt"] = f"1e-{m.group(1)}"

    m = re.search(r"R\$\^2\s*=\s*([0-9.]+)\$\s*for mAP prediction", tex)
    if m:
        out["r2_map"] = float(m.group(1))

    m = re.search(r"On\s*([0-9]+)\s*configurations evaluated at both low", tex)
    if m:
        out["matched_configurations"] = int(m.group(1))

    top5 = re.search(r"Top-5 recall is\s*([0-9.]+)\s*\((\d+)\s*of\s*the\s*top\s*(\d+)\s*", tex)
    if top5:
        out["top5_recall"] = float(top5.group(1))
        out["top5_hits"] = int(top5.group(2))
        out["top5_total"] = int(top5.group(3))

    return out


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    reported = {
        "spearman_rho": None,
        "kendall_tau": None,
        "p_spearman_lt": None,
        "p_kendall_lt": None,
        "r2_map": None,
        "matched_configurations": None,
        "top5_recall": None,
        "top5_hits": None,
        "top5_total": None,
    }
    if PAPER_TEX.exists():
        tex = PAPER_TEX.read_text(encoding="utf-8")
        reported = parse_reported_metrics_from_paper(tex)

    correction_analysis = load_json_if_exists(CORRECTION_ANALYSIS_JSON)
    ablation = load_json_if_exists(ABLATION_JSON)

    raw_corr = None
    if isinstance(ablation, list):
        for row in ablation:
            if row.get("name") in {"− Correction model", "- Correction model"}:
                raw_corr = {
                    "rho_raw": row.get("rho_raw"),
                    "tau_raw": row.get("tau_raw"),
                    "note": row.get("note"),
                }
                break

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "paper_tex": str(PAPER_TEX),
            "correction_analysis_json": str(CORRECTION_ANALYSIS_JSON),
            "ablation_results_json": str(ABLATION_JSON),
        },
        "reported_correction_metrics": reported,
        "supporting_artifacts": {
            "correction_analysis": correction_analysis,
            "ablation_no_correction_raw": raw_corr,
        },
        "notes": [
            "reported_correction_metrics are extracted from paper/main.tex",
            "supporting_artifacts are loaded from experiment output JSON files when present",
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
