import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "phase-3_research-atlas"
    / "vacuum-detector"
    / "algorithm.py"
)

spec = importlib.util.spec_from_file_location("vacuum_detector_algorithm", MODULE_PATH)
vacuum_detector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vacuum_detector
spec.loader.exec_module(vacuum_detector)

BatchVacuumDetector = vacuum_detector.BatchVacuumDetector
Claim = vacuum_detector.Claim
Evidence = vacuum_detector.Evidence
VacuumDetector = vacuum_detector.VacuumDetector
VacuumType = vacuum_detector.VacuumType


def test_missing_evidence_creates_critical_structural_vacuum():
    detector = VacuumDetector()
    claim = Claim(
        id="coordination",
        statement="Vacuum detection improves cross-domain coordination",
        domains=["hokage", "research"],
    )

    vacuums = detector.detect([claim], {})

    assert len(vacuums) == 1
    vacuum = vacuums[0]
    assert vacuum.id == "vac-coordination"
    assert vacuum.type is VacuumType.STRUCTURAL
    assert vacuum.evidence_coverage == 0.0
    assert vacuum.priority == "critical"
    assert vacuum.affected_domains == ["hokage", "research"]
    assert vacuum.recommended_action == "collect_baseline_data"
    datetime.fromisoformat(vacuum.detected_at)


def test_classifies_vacuums_by_the_first_missing_evidence_layer():
    detector = VacuumDetector()
    claims = [
        Claim(
            id="measurement",
            statement="Research logging captures enough observations",
            domains=["research"],
            evidentiary_threshold=1.0,
        ),
        Claim(
            id="inference",
            statement="The detector causes better phase selection",
            domains=["hokage"],
            evidentiary_threshold=1.0,
        ),
        Claim(
            id="attribution",
            statement="Revenue movement can be attributed to research work",
            domains=["commerce"],
            evidentiary_threshold=1.0,
        ),
        Claim(
            id="interface",
            statement="Embedded knowledge is retrievable from every surface",
            domains=["interface"],
            evidentiary_threshold=1.1,
        ),
    ]
    evidence_base = {
        "measurement": Evidence(
            claim_id="measurement",
            has_observations=True,
        ),
        "inference": Evidence(
            claim_id="inference",
            has_observations=True,
            has_statistics=True,
        ),
        "attribution": Evidence(
            claim_id="attribution",
            has_observations=True,
            has_statistics=True,
            has_causal_tests=True,
        ),
        "interface": Evidence(
            claim_id="interface",
            has_observations=True,
            has_statistics=True,
            has_causal_tests=True,
            has_attribution=True,
            sample_size=40,
        ),
    }

    vacuums = detector.detect(claims, evidence_base)

    by_id = {vacuum.id: vacuum for vacuum in vacuums}
    assert by_id["vac-measurement"].type is VacuumType.MEASUREMENT
    assert by_id["vac-measurement"].recommended_action == "add_instrumentation"
    assert by_id["vac-inference"].type is VacuumType.INFERENCE
    assert by_id["vac-inference"].recommended_action == "design_causal_study"
    assert by_id["vac-attribution"].type is VacuumType.ATTRIBUTION
    assert by_id["vac-attribution"].recommended_action == "build_attribution_model"
    assert by_id["vac-interface"].type is VacuumType.INTERFACE
    assert by_id["vac-interface"].recommended_action == "build_embedding_pipeline"


def test_detect_suppresses_claims_when_coverage_meets_threshold():
    detector = VacuumDetector()
    claim = Claim(
        id="supported",
        statement="A fully measured claim should not become a vacuum",
        domains=["research"],
    )
    evidence = Evidence(
        claim_id="supported",
        has_observations=True,
        has_statistics=True,
        has_causal_tests=True,
        has_attribution=True,
        sample_size=31,
    )

    assert detector.detect([claim], {"supported": evidence}) == []


def test_detect_ranks_critical_before_high_and_medium_vacuums():
    detector = VacuumDetector()
    claims = [
        Claim(
            id="medium",
            statement="Multi-domain measured gap",
            domains=["a", "b"],
            evidentiary_threshold=0.8,
        ),
        Claim(
            id="high",
            statement="Single-domain measured gap",
            domains=["a"],
            evidentiary_threshold=0.8,
        ),
        Claim(id="critical", statement="No observations", domains=["a"]),
    ]
    evidence_base = {
        "medium": Evidence(
            claim_id="medium",
            has_observations=True,
            has_statistics=True,
        ),
        "high": Evidence(
            claim_id="high",
            has_observations=True,
            has_statistics=True,
        ),
        "critical": Evidence(claim_id="critical"),
    }

    vacuums = detector.detect(claims, evidence_base)

    assert [vacuum.id for vacuum in vacuums] == [
        "vac-critical",
        "vac-high",
        "vac-medium",
    ]
    assert [vacuum.priority for vacuum in vacuums] == [
        "critical",
        "high",
        "medium",
    ]


def test_vacuum_to_dict_serializes_enum_and_core_fields():
    detector = VacuumDetector()
    claim = Claim(id="capture", statement="Stats are captured", domains=["research"])
    evidence = Evidence(
        claim_id="capture",
        has_observations=True,
        has_statistics=False,
    )

    payload = detector.detect([claim], {"capture": evidence})[0].to_dict()

    assert payload["id"] == "vac-capture"
    assert payload["type"] == "measurement"
    assert payload["claim"] == "Stats are captured"
    assert payload["evidence_coverage"] == 0.0
    assert payload["priority"] == "critical"
    assert payload["affected_domains"] == ["research"]
    assert payload["recommended_action"] == "add_instrumentation"


def test_batch_detector_loads_claims_and_exports_vacuums(tmp_path):
    claims_file = tmp_path / "claims.json"
    evidence_file = tmp_path / "evidence.json"
    output_file = tmp_path / "vacuums.json"

    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "id": "causal-gap",
                        "statement": "Routing quality improves after detector use",
                        "domains": ["hokage"],
                        "evidentiary_threshold": 0.8,
                    }
                ]
            }
        )
    )
    evidence_file.write_text(
        json.dumps(
            {
                "evidence": {
                    "causal-gap": {
                        "claim_id": "causal-gap",
                        "has_observations": True,
                        "has_statistics": True,
                        "has_causal_tests": False,
                        "sample_size": 45,
                    }
                }
            }
        )
    )
    batch_detector = BatchVacuumDetector(
        config={
            "evidence_threshold": 0.2,
            "causal_weight": 0.4,
            "measurement_weight": 0.3,
            "attribution_weight": 0.3,
        }
    )

    vacuums = batch_detector.detect_from_file(str(claims_file), str(evidence_file))
    batch_detector.export_vacuums(vacuums, str(output_file))

    exported = json.loads(output_file.read_text())
    assert len(vacuums) == 1
    assert vacuums[0].type is VacuumType.INFERENCE
    assert exported["count"] == 1
    assert exported["vacuums"][0]["id"] == "vac-causal-gap"
    assert exported["vacuums"][0]["type"] == "inference"
    datetime.fromisoformat(exported["exported_at"])
