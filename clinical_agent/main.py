from clinical_agent.amplifier import AmplifierClient
from clinical_agent.api import create_app
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber

settings = Settings.from_env()
store = PatientStore(settings.data_dir)
bus = EventBus()
if not store.list_patients():
    generate_synthetic_patient(store)
transcriber = Transcriber(settings.whisper_model, mock=settings.mock_whisper)
amplifier = AmplifierClient(settings, bus)
app = create_app(settings, store, bus, transcriber, amplifier)
