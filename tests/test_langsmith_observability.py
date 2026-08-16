"""LangSmith 初期接入测试。"""

import os

from app.observability import langsmith as langsmith_observability


def test_configure_langsmith_exports_validated_settings(monkeypatch):
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_TRACING_SAMPLING_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    langsmith_observability.configure_langsmith()

    assert os.environ["LANGSMITH_TRACING"] in {"true", "false"}
    assert os.environ["LANGSMITH_PROJECT"] == (
        langsmith_observability.settings.langsmith_project
    )
    assert os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] == str(
        langsmith_observability.settings.langsmith_sampling_rate
    )


def test_build_agent_trace_config_contains_business_context():
    config = langsmith_observability.build_agent_trace_config("conversation-123")

    assert config["run_name"] == "zhixing_travel_agent"
    assert "agent:travel" in config["tags"]
    assert config["metadata"]["conversation_id"] == "conversation-123"
    assert config["metadata"]["model"]
    assert config["configurable"]["thread_id"] == "conversation-123"


def test_flush_waits_for_cached_tracers_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(langsmith_observability.settings, "langsmith_tracing", True)
    monkeypatch.setattr(
        langsmith_observability,
        "wait_for_all_tracers",
        lambda: calls.append("flushed"),
    )

    langsmith_observability.flush_langsmith_traces()

    assert calls == ["flushed"]


def test_flush_is_skipped_when_tracing_is_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(langsmith_observability.settings, "langsmith_tracing", False)
    monkeypatch.setattr(
        langsmith_observability,
        "wait_for_all_tracers",
        lambda: calls.append("flushed"),
    )

    langsmith_observability.flush_langsmith_traces()

    assert calls == []


def test_flush_failure_does_not_block_shutdown(monkeypatch):
    monkeypatch.setattr(langsmith_observability.settings, "langsmith_tracing", True)

    def fail_flush():
        raise RuntimeError("upload failed")

    monkeypatch.setattr(
        langsmith_observability,
        "wait_for_all_tracers",
        fail_flush,
    )

    # 上传异常应被记录，但不能中断 FastAPI 的关闭流程。
    langsmith_observability.flush_langsmith_traces()
