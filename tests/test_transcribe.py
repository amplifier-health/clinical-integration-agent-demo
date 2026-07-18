from clinical_agent.transcribe import Transcriber


async def test_mock_transcriber():
    t = Transcriber(mock=True)
    assert await t.transcribe(b"anything") == "[mock transcript]"


async def test_real_model_is_lazy():
    t = Transcriber(mock=False)
    assert t._model is None  # no model download at construction time
