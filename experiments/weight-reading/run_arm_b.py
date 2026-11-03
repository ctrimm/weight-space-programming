"""Arm B — behavior rescue for reads with R2 < 1 (only the two
trained-control reads qualified). Each reader gets 25 input/output
samples from the mystery network plus its own Arm A answer, and may
revise. Same answer schema. Cost logged; same $6 cumulative cap
(shared cost_log.json with Arm A).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response  # noqa: E402

HERE = Path(__file__).parent
CAP_USD = 6.00
MAX_TOKENS = 24_000
CASES = [("openai/gpt-5.6-sol", "gpt-5.6-sol", "N11"),
         ("google/gemini-3.1-pro-preview", "gemini-3.1-pro-preview", "N11")]

sys.path.insert(0, str(HERE.parent / "ssi-payment"))
import mlp  # noqa: E402


def samples(code: str) -> str:
    w = mlp.load_weights(str(HERE / "mystery" / f"{code}.npz"))
    Es = [0, 85, 500, 1000, 2073, 3000]
    Us = [0, 20, 500, 1014]
    pts = [(e, u) for e in Es for u in Us] + [(1234, 77)]
    E = np.array([p[0] for p in pts], float)
    U = np.array([p[1] for p in pts], float)
    out = mlp.ssi_forward(w, E, U)
    lines = ["E (earned $), U (unearned $) -> network output ($)"]
    for (e, u), o in zip(pts, out):
        lines.append(f"E={e}, U={u} -> {o:.2f}")
    return "\n".join(lines)


def call_api(model: str, prompt: str) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "usage": {"include": True}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "HTTP-Referer": "https://github.com/ctrimm/weight-space-programming",
                 "X-Title": "weight-space-programming weight-reading arm B"})
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    with urllib.request.urlopen(req, timeout=900) as resp:
        return read_json_response(resp)


def main() -> None:
    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text())
    spent = float(log["total_usd"])
    for model, mslug, code in CASES:
        raw_path = HERE / "raw" / f"{mslug}__{code}__armB.txt"
        if raw_path.exists():
            print(f"skip (done): {model} {code}", flush=True)
            continue
        if spent >= CAP_USD:
            print(f"HARD CAP reached (${spent:.4f}); aborting Arm B for {model} {code}", flush=True)
            return
        arm_a = json.loads((HERE / "claims" / f"{mslug}__{code}__armA.json").read_text())
        prompt = (
            "You previously analyzed a neural network's weights (inputs: E = monthly earned "
            "income, U = monthly unearned income, in dollars; output: a monthly payment in "
            "dollars) and answered:\n\n" + json.dumps(arm_a, indent=2) +
            "\n\nYour claimed formula does not exactly reproduce the network's behavior. "
            "Here are 25 measured input/output samples from the network itself:\n\n"
            + samples(code) +
            "\n\nRevise your answer. Reply with ONE JSON object in the same schema: "
            '{"program": ..., "formula_text": ..., "constants": {...}, '
            '"formula_python": "<ONE Python expression over E and U using only numbers, max, min, abs, +, -, *, /, parentheses>", '
            '"confidence": ..., "anomalies": [...]}. '
            "The revised formula_python will be checked against the network on tens of thousands of inputs; "
            "make it match the network's actual behavior as closely as a single expression can, "
            "not merely the textbook rule the network approximates.")
        t0 = time.time()
        try:
            resp = call_api(model, prompt)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} {model} {code} armB: {e.read().decode()[:300]}", flush=True)
            continue
        usage = resp.get("usage", {}) or {}
        cost = float(usage.get("cost", 0.0) or 0.0)
        spent += cost
        log["total_usd"] = spent
        text = resp["choices"][0]["message"]["content"] or ""
        raw_path.write_text(text)
        entry = {"model": model, "artifact": code, "arm": "B", "cost_usd": cost,
                 "prompt_tokens": usage.get("prompt_tokens"),
                 "completion_tokens": usage.get("completion_tokens"),
                 "seconds": round(time.time() - t0, 1)}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            claim = json.loads(m.group(0)) if m else (_ for _ in ()).throw(ValueError("no JSON"))
            (HERE / "claims" / f"{mslug}__{code}__armB.json").write_text(json.dumps(claim, indent=2))
            entry["parse"] = "ok"
            print(f"{model} {code} armB: parsed, ${cost:.4f} (total ${spent:.4f})", flush=True)
        except Exception as exc:
            entry["parse"] = f"failed: {exc}"
            print(f"{model} {code} armB: PARSE FAILED ({exc}), ${cost:.4f}", flush=True)
        log["calls"].append(entry)
        log_path.write_text(json.dumps(log, indent=2))
    print(f"ARM B DONE. total measured spend: ${spent:.4f} of ${CAP_USD} cap", flush=True)


if __name__ == "__main__":
    main()
