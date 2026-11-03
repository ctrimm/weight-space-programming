"""Build the anonymized mystery pack: copy the eleven source networks
under neutral codes, hash-freeze them, and record the sealed ground
truth mapping (never sent to any reader; prompts are generated from
artifact files + framing text only)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

SOURCES = {
    "N01": ("experiments/snap-allotment/zero_shot/snap_weights.npz", "snap", "claude"),
    "N02": ("experiments/second-writer-openrouter/weights/gpt-5.6-sol__snap.npz", "snap", "gpt"),
    "N03": ("experiments/second-writer-openrouter/weights/gemini-3.1-pro-preview__snap.npz", "snap", "gemini"),
    "N04": ("experiments/ssi-payment/zero_shot/ssi_weights.npz", "ssi", "claude"),
    "N05": ("experiments/second-writer-openrouter/weights/gpt-5.6-sol__ssi.npz", "ssi", "gpt"),
    "N06": ("experiments/second-writer-openrouter/weights/gemini-3.1-pro-preview__ssi.npz", "ssi", "gemini"),
    "N07": ("experiments/medicaid-patient-liability/zero_shot/medicaid_weights.npz", "medicaid", "claude"),
    "N08": ("experiments/second-writer-openrouter/weights/gpt-5.6-sol__medicaid.npz", "medicaid", "gpt"),
    "N09": ("experiments/second-writer-openrouter/weights/gemini-3.1-pro-preview__medicaid.npz", "medicaid", "gemini"),
    "N10": ("experiments/ssi-stale-fbr-edit/stale/ssi_stale_weights.npz", "ssi_defect", "claude"),
    "N11": ("experiments/ssi-payment/trained/ssi_weights.npz", "ssi_control_trained", "trained"),
}

ASSIGNMENTS = {
    "openai/gpt-5.6-sol": ["N03", "N04", "N09", "N10", "N11"],
    "google/gemini-3.1-pro-preview": ["N01", "N05", "N07", "N10", "N11"],
    "claude-in-session": ["N02", "N06", "N08"],
}


def main() -> None:
    truth, freeze = {}, {}
    for code, (rel, family, writer) in SOURCES.items():
        src = ROOT / rel
        dst = HERE / "mystery" / f"{code}.npz"
        shutil.copyfile(src, dst)
        freeze[f"mystery/{code}.npz"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        truth[code] = {"family": family, "writer": writer, "source": rel}
    (HERE / "ground_truth.json").write_text(json.dumps(
        {"artifacts": truth, "assignments": ASSIGNMENTS}, indent=2))
    (HERE / "FREEZE.json").write_text(json.dumps(
        {"frozen_before_reads": True, "files": freeze}, indent=2))
    print(f"packed {len(SOURCES)} artifacts; assignments:",
          {k: len(v) for k, v in ASSIGNMENTS.items()})


if __name__ == "__main__":
    main()
