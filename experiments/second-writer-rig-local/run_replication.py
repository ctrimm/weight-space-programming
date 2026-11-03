"""Second-writer replication on local open-weight models (the rig).

Calls `q38` (Qwen3.8-27B) and `strata` (Qwen3.8-Flash-Next) through
the homelab LiteLLM gateway, one call per writer x task, with at most
one syntax-repair retry per cell. Prompts are not retyped: they are
built by importing the three earlier second-writer runners, and the
witness-extension and SNAP-composition prompts are checked against
those runs' frozen hashes before any call.

Credentials: the gateway key is read from the RIG_LITELLM_KEY
environment variable; it is never written to disk or printed here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent
BASE_URL = os.environ.get("RIG_LITELLM_URL", "https://llm.tabaska.us/v1")
MAX_TOKENS = 32_768
TIMEOUT_S = 3_600
TRANSPORT_RETRIES = 3
MODELS = ["q38", "strata"]

# The earlier runners import an OpenRouter credential helper at module
# load; stub it so their TASKS/FORMAT_SPEC can be imported verbatim.
_stub = types.ModuleType("dynamic_credentials")
_stub.add_surrogate_to_request = lambda *a, **k: None
_stub.read_json_response = lambda resp: json.load(resp)
sys.modules["dynamic_credentials"] = _stub


def load_runner(folder: str, filename: str):
    path = EXP / folder / filename
    spec = importlib.util.spec_from_file_location(f"runner_{folder.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOURCES = {
    "second-writer-openrouter": ("run_pilot.py", ["eitc", "snap", "ssi", "medicaid"]),
    "second-writer-witness-extension": ("run_replication.py", ["soc", "couple"]),
    "second-writer-snap-composition": ("run_replication.py", ["snapcomp"]),
}
ORDER = ["eitc", "snap", "ssi", "medicaid", "soc", "couple", "snapcomp"]

TASKS: dict[str, dict] = {}
for folder, (filename, tasks) in SOURCES.items():
    runner = load_runner(folder, filename)
    for task in tasks:
        TASKS[task] = {
            "source": folder,
            "prompt": runner.TASKS[task]["rule"] + "\n" + runner.FORMAT_SPEC.format(arch=runner.TASKS[task]["arch"]),
            "arch": runner.TASKS[task]["arch"],
            "shapes": runner.TASKS[task].get("shapes"),
        }


def shapes_for(task: str) -> dict[str, tuple[int, ...]]:
    """Fixed array shapes; the first pilot only states them in the arch text."""
    if TASKS[task]["shapes"]:
        return {k: tuple(v) for k, v in TASKS[task]["shapes"].items()}
    return {name: tuple(int(d) for d in dims.split(","))
            for name, dims in re.findall(r"\b([Wb]\d) \[([\d,]+)\]", TASKS[task]["arch"])}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_prompt_freeze() -> None:
    for task in ORDER:
        frozen = EXP / TASKS[task]["source"] / "results" / "prompt_freeze.json"
        if frozen.exists():
            want = json.loads(frozen.read_text())["prompts"][task]["sha256"]
            got = sha256_bytes(TASKS[task]["prompt"].encode("utf-8"))
            if got != want:
                raise SystemExit(f"prompt for {task} does not match {frozen}: {got} != {want}")
    path = HERE / "results" / "prompt_freeze.json"
    if path.exists():
        return
    path.write_text(json.dumps({
        "created_before_first_model_call": True,
        "runner_file": "run_replication.py",
        "prompts": {task: {"sha256": sha256_bytes(TASKS[task]["prompt"].encode("utf-8")),
                           "characters": len(TASKS[task]["prompt"]),
                           "source": TASKS[task]["source"]} for task in ORDER},
    }, indent=2))


def call_api(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {os.environ['RIG_LITELLM_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def call_with_transport_retries(model: str, prompt: str, transport_log: list) -> dict:
    for n in range(TRANSPORT_RETRIES + 1):
        try:
            return call_api(model, prompt)
        except urllib.error.HTTPError as exc:
            if exc.code not in (502, 503, 504) or n == TRANSPORT_RETRIES:
                raise
            transport_log.append(f"HTTP {exc.code}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if n == TRANSPORT_RETRIES:
                raise
            transport_log.append(f"{type(exc).__name__}: {exc}")
        print(f"  transport retry {n + 1}/{TRANSPORT_RETRIES} for {model}: {transport_log[-1]}", flush=True)
        time.sleep(30)
    raise RuntimeError("unreachable")


def parse_weights(text: str, task: str) -> dict[str, np.ndarray]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in reply")
    spec = json.loads(match.group(0))
    arrays = spec["arrays"]
    out: dict[str, np.ndarray] = {}
    for name, shape in shapes_for(task).items():
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
    records = {p.name: {"sha256": sha256_bytes(p.read_bytes()), "bytes": p.stat().st_size}
               for p in sorted((HERE / "weights").glob("*.npz"))}
    (HERE / "weights" / "FREEZE.json").write_text(json.dumps(records, indent=2))


def main() -> None:
    for sub in ("raw", "weights", "results"):
        (HERE / sub).mkdir(parents=True, exist_ok=True)
    write_prompt_freeze()

    log_path = HERE / "results" / "cost_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"max_tokens": MAX_TOKENS, "calls": []}
    done = {(c["model"], c["task"]) for c in log["calls"] if c.get("final")}

    for model in MODELS:
        for task in ORDER:
            dest = HERE / "weights" / f"{model}__{task}.npz"
            if dest.exists() or (model, task) in done:
                print(f"skip (done): {model} {task}", flush=True)
                continue
            prompt = TASKS[task]["prompt"]
            for attempt in (1, 2):
                started = time.time()
                transport: list[str] = []
                entry = {"model": model, "task": task, "attempt": attempt}
                try:
                    response = call_with_transport_retries(model, prompt, transport)
                except Exception as exc:
                    entry.update(error=f"{type(exc).__name__}: {exc}", transport_retries=transport,
                                 seconds=round(time.time() - started, 1), final=True)
                    print(f"{model} {task} attempt {attempt}: ERROR {entry['error']}", flush=True)
                    log["calls"].append(entry)
                    log_path.write_text(json.dumps(log, indent=2))
                    break

                usage = response.get("usage", {}) or {}
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                raw_path = HERE / "raw" / f"{model}__{task}__attempt{attempt}.txt"
                raw_path.write_text(text)
                if reasoning:
                    (HERE / "raw" / f"{model}__{task}__attempt{attempt}.reasoning.txt").write_text(reasoning)
                entry.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                             finish_reason=choice.get("finish_reason"), seconds=round(time.time() - started, 1),
                             transport_retries=transport, raw_file=raw_path.name)
                try:
                    weights = parse_weights(text, task)
                except Exception as exc:
                    entry["parse"] = f"failed: {exc}"
                    entry["final"] = attempt == 2
                    print(f"{model} {task} attempt {attempt}: PARSE FAILED ({exc}), "
                          f"{entry['completion_tokens']} tok, {entry['seconds']}s, finish={entry['finish_reason']}", flush=True)
                    log["calls"].append(entry)
                    log_path.write_text(json.dumps(log, indent=2))
                    prompt = (TASKS[task]["prompt"]
                              + f"\nIMPORTANT: a previous reply could not be parsed ({exc}). "
                                "Reply with ONLY the JSON object, no other text.")
                    continue

                np.savez(dest, **weights)
                entry.update(parse="ok", final=True,
                             nonzero=int(sum(int(np.count_nonzero(v)) for v in weights.values())))
                print(f"{model} {task} attempt {attempt}: parsed, {entry['nonzero']} nonzero, "
                      f"{entry['completion_tokens']} tok, {entry['seconds']}s", flush=True)
                log["calls"].append(entry)
                log_path.write_text(json.dumps(log, indent=2))
                break

    freeze_weights()
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
