from pathlib import Path

from clinical_agent import gcs


def test_local_path_passthrough(tmp_path):
    p = tmp_path / "x"; p.write_text("hi")
    assert gcs.localize(p) == Path(p)
    assert gcs.localize(str(p)) == Path(p)


def test_gs_uri_shells_out(tmp_path, monkeypatch):
    calls = []

    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **k):
        calls.append(cmd)
        (tmp_path / "aggregate.json").write_text("{}")  # simulate the download landing
        return R()

    monkeypatch.setattr(gcs.subprocess, "run", fake_run)
    out = gcs.localize("gs://bucket/demo/aggregate.json", dest=tmp_path)
    assert calls and calls[0][0] in ("gcloud", "gsutil") and "gs://bucket/demo/aggregate.json" in calls[0]
    assert out == tmp_path / "aggregate.json"
