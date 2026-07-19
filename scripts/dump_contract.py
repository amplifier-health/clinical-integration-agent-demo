"""Render the output contract (clinical_agent/contract.py) to JSON Schema + a
machine-readable index in docs/contract/. This is the artifact a plugin consumer
integrates against. Regenerate after any model change:

    python scripts/dump_contract.py

The schema-current test (tests/test_contract.py) fails if the committed files
drift from the models, so a contract change can't merge without regenerating.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_agent import contract  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "contract"


def build() -> dict:
    events = {}
    for type_name, (model, phase) in sorted(contract.REGISTRY.items()):
        events[type_name] = {
            "phase": phase,
            "clinical": type_name in contract.CLINICAL_TYPES,
            "label": contract.label_for(type_name),
            "description": contract.description_for(type_name),
            "schema": model.model_json_schema(),
        }
    return {
        "contract_version": contract.CONTRACT_VERSION,
        "envelope_fields": ["type", "contract_version", "phase", "session_id", "seq", "ts"],
        "wire_format": "flat — envelope fields and payload fields share the top level",
        "tiers": list(contract.TIER.__args__),
        "clinical_types": sorted(contract.CLINICAL_TYPES),
        "events": events,
    }


def write() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract.json").write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    write()
    print(f"wrote {OUT / 'contract.json'} ({len(contract.REGISTRY)} event types, v{contract.CONTRACT_VERSION})")
