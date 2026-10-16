"""Usage survives retries, rejected output and an unfinished request, without secrets."""

import json
from pathlib import Path

from benchmark.causal_memory.lenovo_gemma_provider import LenovoGemmaProvider
from benchmark.causal_memory.model_usage import ModelUsageRecorder, normalized_usage, read_usage
from openviking.session.aig_toolloop_builder import ToolloopTurn


class Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_retry_and_rejected_output_are_accounted_before_report(tmp_path: Path):
    calls = 0

    def transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("secret-shaped diagnostics must not be logged")
        return Response(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "wrong output"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            }
        )

    path = tmp_path / "usage.jsonl"
    provider = LenovoGemmaProvider(
        api_key="never-log-this-key",
        model="GLM-5.3-Flash",
        urlopen=transport,
        sleep=lambda _: None,
        usage_path=path,
    )
    provider.bind_audit_context(role="critic", run_id="episode-a2")
    assert provider(ToolloopTurn("private prompt", (), ())).verb == "invalid_provider_result"
    report = read_usage(path)
    total = report["overall"]
    assert total["http_attempts_started"] == total["http_attempts_completed"] == 2
    assert total["logical_completions"] == 1
    assert total["network_retry_attempts"] == 1
    assert total["known_total_tokens"] == 120
    assert total["known_cached_input_tokens"] == 40
    assert total["known_reasoning_tokens"] == 5
    assert total["usage_complete"] is False  # Failed request usage is unknown.
    assert total["validation_status_counts"] == {"invalid_provider_result": 1}
    assert report["by_run"]["episode-a2"]["by_role"]["critic"] == total
    log = path.read_text()
    assert "private prompt" not in log and "never-log-this-key" not in log
    assert "secret-shaped" not in log


def test_missing_usage_and_killed_request_are_not_counted_as_zero(tmp_path: Path):
    path = tmp_path / "usage.jsonl"
    recorder = ModelUsageRecorder(path)
    recorder.bind(role="reader", run_id="episode")
    recorder.begin_completion()
    request_id, started = recorder.started(model="GLM-5.3-Flash", network_attempt=0)
    # The start is visible before any response or final four-role report exists.
    assert read_usage(path)["overall"]["incomplete_http_attempts"] == 1
    recorder.completed(
        request_id=request_id,
        started=started,
        model="GLM-5.3-Flash",
        network_attempt=0,
        status="ok",
        http_status=200,
        payload={},
    )
    total = read_usage(path)["overall"]
    assert total["incomplete_http_attempts"] == 0
    assert total["completed_attempts_without_full_usage"] == 1
    assert total["usage_complete"] is False
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["usage"]["input_tokens"] is None
    assert row["usage"]["reported"] is False


def test_usage_normalization_preserves_zero_and_does_not_double_count_reasoning():
    usage = normalized_usage(
        {
            "usage": {
                "input_tokens": 0,
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 10},
            }
        }
    )
    assert usage["input_tokens"] == 0
    assert usage["total_tokens"] == 12
    assert usage["reasoning_tokens"] == 10
    assert usage["reported"] is True
    assert normalized_usage({"usage": {"prompt_tokens": True}})["input_tokens"] is None
