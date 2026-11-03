"""Second-writer replication runner for the witness-extension rules.

Calls GPT-5.6 Sol and Gemini 3.1 Pro through OpenRouter, one call per
model x task, with at most one syntax-repair retry per cell. Actual
per-call costs come from each response's usage accounting. The runner
checks cumulative measured spend before every call and stops before
another call once the $0.75 hard cap in PROTOCOL.md has been reached.

Credentials are injected through the custom.openrouter surrogate;
no credential value is read or printed here.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response  # noqa: E402

HERE = Path(__file__).parent
CAP_USD = 0.75
MAX_TOKENS = 20_000
MODELS = ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview"]

FORMAT_SPEC = """
NETWORK SPEC (fixed architecture; scaling is part of the architecture, not the weights):
- Forward pass: for each layer i, h = W_i @ h_prev + b_i, followed by ReLU after every layer except the last. W_i has shape [out_features, in_features]; entry "r,c" multiplies input component c into output unit r.
- {arch}
- Return the weights as ONE JSON object, no markdown, no commentary:
  {{"arrays": {{"W1": {{"shape": [rows, cols], "entries": {{"r,c": number, ...}}}}, "b1": {{"shape": [n], "entries": {{"i": number, ...}}}}, ...}}}}
  List ONLY nonzero entries; every unlisted entry is 0. Include every array named in the architecture, even if all its entries are 0 (then "entries": {{}}).
- Work in the scaled units the architecture defines. Derive the weights by reasoning about the formula. Do not approximate: the network will be checked against the formula on tens of thousands of inputs and must match to the cent everywhere.
"""

TASKS = {
    "soc": {
        "arch": "Inputs: [E/1000, U/1000, M/1000] (E = monthly earned income, U = monthly unearned income, M = monthly incurred medical expenses, dollars). Layers: W1 [24,3], b1 [24], then ReLU, then W2 [24,24], b2 [24], then ReLU, then W3 [24,24], b3 [24], then ReLU, then W4 [1,24], b4 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2, W3, b3, W4, b4.",
        "shapes": {
            "W1": (24, 3), "b1": (24,), "W2": (24, 24), "b2": (24,),
            "W3": (24, 24), "b3": (24,), "W4": (1, 24), "b4": (1,),
        },
        "rule": """FORMULA (California Medicaid medically-needy share of cost, individual, monthly):
- Countable unearned income CU = max(0, U - 20)  (the $20 general exclusion applies to unearned income first)
- Unused exclusion spills to earned income: R = max(0, 20 - U)
- Countable earned income CE = max(0, E - 65 - R) / 2  ($65 exclusion plus the spill, then one-half disregarded)
- Countable income C = CU + CE
- Share of cost SOC = max(0, C - 600 - M)  ($600 maintenance-need income level, then subtract incurred medical expenses M)
Domain: E in [0, 3000], U in [0, 3000], M in [0, 2000]. No eligibility gates, no premium deduction, no rounding.""",
    },
    "couple": {
        "arch": "Inputs: [E1/1000, U1/1000, E2/1000, U2/1000] (E1 and U1 are member 1's monthly earned and unearned income; E2 and U2 are member 2's monthly earned and unearned income, dollars). Layers: W1 [24,4], b1 [24], then ReLU, then W2 [24,24], b2 [24], then ReLU, then W3 [24,24], b3 [24], then ReLU, then W4 [1,24], b4 [1] (final layer linear). Output dollars = 1000 * (final linear output). Arrays: W1, b1, W2, b2, W3, b3, W4, b4.",
        "shapes": {
            "W1": (24, 4), "b1": (24,), "W2": (24, 24), "b2": (24,),
            "W3": (24, 24), "b3": (24,), "W4": (1, 24), "b4": (1,),
        },
        "rule": """FORMULA (US Supplemental Security Income federal monthly payment, eligible couple, 2026):
- First combine the two members' incomes: E = E1 + E2 and U = U1 + U2
- Countable unearned income CU = max(0, U - 20)  (the $20 general exclusion applies once, to the combined unearned income)
- Unused exclusion spills to earned income: R = max(0, 20 - U)
- Countable earned income CE = max(0, E - 65 - R) / 2  ($65 exclusion once, plus the spill, then one-half disregarded)
- Countable income C = CU + CE
- Payment P = max(0, 1491 - C)  ($1,491 federal benefit rate for a couple)
Domain: E1 in [0, 2000], U1 in [0, 1500], E2 in [0, 2000], U2 in [0, 1500]. Federal payment only, no rounding.""",
    },
}
ORDER = ["soc", "couple"]


def prompt_for(task: str) -> str:
    return TASKS[task]["rule"] + "\n" + FORMAT_SPEC.format(arch=TASKS[task]["arch"])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_prompt_freeze() -> None:
    path = HERE / "results" / "prompt_freeze.json"
    if path.exists():
        return
    prompts = {}
    for task in ORDER:
        prompt = prompt_for(task)
        prompts[task] = {
            "sha256": sha256_bytes(prompt.encode("utf-8")),
            "characters": len(prompt),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_before_first_model_call": True,
        "runner_file": "run_replication.py",
        "prompts": prompts,
    }, indent=2))


def signed_request(url: str, payload: dict | None = None) -> urllib.request.Request:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        method="GET" if payload is None else "POST",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/ctrimm/weight-space-programming",
            "X-Title": "weight-space-programming second-writer witness extension",
        },
    )
    add_surrogate_to_request(req, "custom.openrouter", allowed_hosts=["openrouter.ai"])
    return req


def get_account_usage() -> dict:
    with urllib.request.urlopen(
        signed_request("https://openrouter.ai/api/v1/credits"), timeout=60
    ) as resp:
        data = read_json_response(resp)["data"]
    total_credits = float(data["total_credits"])
    total_usage = float(data["total_usage"])
    return {
        "total_credits_usd": total_credits,
        "total_usage_usd": total_usage,
        "balance_usd": total_credits - total_usage,
    }


def record_account_usage(stage: str) -> None:
    path = HERE / "results" / "account_usage.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if stage in saved:
        return
    try:
        saved[stage] = {**get_account_usage(), "recorded_at_unix": time.time()}
    except Exception as exc:  # Account checks must not be silently treated as spend data.
        saved[stage] = {"error": f"{type(exc).__name__}: {exc}", "recorded_at_unix": time.time()}
    path.write_text(json.dumps(saved, indent=2))


def call_api(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    req = signed_request("https://openrouter.ai/api/v1/chat/completions", payload)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return read_json_response(resp)


def parse_weights(text: str, task: str) -> dict[str, np.ndarray]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in reply")
    spec = json.loads(match.group(0))
    arrays = spec["arrays"]
    out: dict[str, np.ndarray] = {}
    for name, shape in TASKS[task]["shapes"].items():
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        returned_shape = tuple(int(x) for x in arrays[name]["shape"])
        if returned_shape != shape:
            raise ValueError(f"wrong shape for {name}: {returned_shape}, expected {shape}")
        arr = np.zeros(shape, dtype=np.float64)
        entries = arrays[name].get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError(f"entries for {name} is not an object")
        for key, val in entries.items():
            idx = tuple(int(part) for part in str(key).split(","))
            if len(idx) != len(shape):
                raise ValueError(f"wrong index rank for {name}: {key}")
            if any(i < 0 or i >= dim for i, dim in zip(idx, shape)):
                raise ValueError(f"index out of bounds for {name}: {key}")
            arr[idx] = float(val)
        out[name] = arr
    return out


def freeze_weights() -> None:
    weights_dir = HERE / "weights"
    records = {}
    for path in sorted(weights_dir.glob("*.npz")):
        records[path.name] = {
            "sha256": sha256_bytes(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    (weights_dir / "FREEZE.json").write_text(json.dumps(records, indent=2))


def main() -> None:
    (HERE / "raw").mkdir(parents=True, exist_ok=True)
    (HERE / "weights").mkdir(parents=True, exist_ok=True)
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    write_prompt_freeze()
    record_account_usage("before")

    log_path = HERE / "results" / "cost_log.json"
    log = (
        json.loads(log_path.read_text())
        if log_path.exists()
        else {"cap_usd": CAP_USD, "calls": [], "total_usd": 0.0}
    )
    spent = float(log["total_usd"])

    for model in MODELS:
        model_slug = model.split("/")[-1]
        for task in ORDER:
            dest = HERE / "weights" / f"{model_slug}__{task}.npz"
            if dest.exists():
                print(f"skip (done): {model} {task}", flush=True)
                continue

            prompt = prompt_for(task)
            for attempt in (1, 2):
                if spent >= CAP_USD:
                    print(
                        f"HARD CAP ${CAP_USD:.2f} reached (${spent:.4f}); "
                        f"aborting before {model} {task} attempt {attempt}",
                        flush=True,
                    )
                    log_path.write_text(json.dumps(log, indent=2))
                    freeze_weights()
                    record_account_usage("after")
                    return

                started = time.time()
                try:
                    response = call_api(model, prompt)
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode(errors="replace")[:500]
                    print(f"HTTP {exc.code} for {model} {task} attempt {attempt}: {body}", flush=True)
                    log["calls"].append({
                        "model": model,
                        "task": task,
                        "attempt": attempt,
                        "error": f"HTTP {exc.code}: {body}",
                        "cost_usd": 0.0,
                    })
                    log_path.write_text(json.dumps(log, indent=2))
                    break
                except Exception as exc:
                    print(f"ERROR for {model} {task} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
                    log["calls"].append({
                        "model": model,
                        "task": task,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cost_usd": 0.0,
                    })
                    log_path.write_text(json.dumps(log, indent=2))
                    break

                usage = response.get("usage", {}) or {}
                cost = float(usage.get("cost", 0.0) or 0.0)
                spent += cost
                log["total_usd"] = spent

                choices = response.get("choices") or []
                text = ""
                if choices:
                    text = (choices[0].get("message") or {}).get("content") or ""
                raw_path = HERE / "raw" / f"{model_slug}__{task}__attempt{attempt}.txt"
                raw_path.write_text(text)

                entry = {
                    "model": model,
                    "task": task,
                    "attempt": attempt,
                    "cost_usd": cost,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "seconds": round(time.time() - started, 1),
                    "raw_file": raw_path.name,
                }
                try:
                    weights = parse_weights(text, task)
                except Exception as exc:
                    entry["parse"] = f"failed: {exc}"
                    print(
                        f"{model} {task} attempt {attempt}: PARSE FAILED ({exc}), "
                        f"${cost:.4f} (total ${spent:.4f})",
                        flush=True,
                    )
                    log["calls"].append(entry)
                    log_path.write_text(json.dumps(log, indent=2))
                    if attempt == 1:
                        prompt = (
                            prompt_for(task)
                            + f"\nIMPORTANT: a previous reply could not be parsed ({exc}). "
                              "Reply with ONLY the JSON object, no other text."
                        )
                    continue

                np.savez(dest, **weights)
                entry["parse"] = "ok"
                entry["nonzero"] = int(sum(int(np.count_nonzero(v)) for v in weights.values()))
                print(
                    f"{model} {task} attempt {attempt}: parsed, {entry['nonzero']} nonzero, "
                    f"${cost:.4f} (total ${spent:.4f})",
                    flush=True,
                )
                log["calls"].append(entry)
                log_path.write_text(json.dumps(log, indent=2))
                break

    log_path.write_text(json.dumps(log, indent=2))
    freeze_weights()
    record_account_usage("after")
    print(f"DONE. total measured spend: ${spent:.4f} of ${CAP_USD:.2f} cap", flush=True)


if __name__ == "__main__":
    main()
