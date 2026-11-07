"""Integration test: run the Inspect eval end-to-end with mockllm and
assert all 16 samples reproduce the detection matrix measured by the
original audit (../results/audit.json) exactly.

No LLM/API calls: the solver runs audit.py's instruments directly
and the model is mockllm/model. $0 spend.
"""
from inspect_ai import eval as inspect_eval

import tamper_audit_task as tat


def test_eval_reproduces_detection_matrix(tmp_path):
    logs = inspect_eval(
        tat.weight_tamper_audit(),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.samples is not None
    assert len(log.samples) == 16

    expected = tat.load_expected_matrix()
    seen = {}
    for sample in log.samples:
        cond, inst = sample.id.split("::")
        score = sample.scores["detection_scorer"]
        # the scorer's verdict: solver's detects matched the matrix
        assert score.value == "C", f"{sample.id}: {score.explanation}"
        # and the detects flag recorded in the log is the matrix cell
        assert score.metadata["detects"] is expected[cond][inst]
        assert score.metadata["expected_detects"] is expected[cond][inst]
        # the browsable record: measurements ride along in the log
        assert "measurements" in score.metadata
        # the solver's JSON record is the sample output
        assert f'"instrument": "{inst}"' in (sample.output.completion or "")
        seen[(cond, inst)] = score.metadata["detects"]

    assert seen == {(c, i): expected[c][i]
                    for c in tat.CONDITIONS for i in tat.INSTRUMENTS}
