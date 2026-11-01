"""Governance, normalization, and artifact tests for Phase 8R6A."""
from __future__ import annotations

import json
import pandas as pd

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.extension.external_normalization import digest, normalize_address, normalize_hex, normalize_label, normalize_opcode
from src.extension.external_overlap import address_summary, add_hashes
from src.extension.external_provenance import complete_benchmark, ground_truth_eligible, track_b_rating
from src.utils.paths import REPO_ROOT

OUT=REPO_ROOT/"results/extension/phase8r6a"


def test_frozen_phase0_to_phase9a_anchors_and_raw_hash():
    observed=verify_frozen_manifest(REPO_ROOT/"configs/extension/frozen_phase9a_manifest.yaml")
    assert len(observed)==9
    assert observed["data/raw/PonziDataset_20221114.csv"]=="48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_normalization_and_hashing_are_deterministic():
    assert normalize_address(" 0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD ")=="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    assert normalize_address("bad") is None
    assert normalize_hex(" 0x60 60\n40 ")=="606040"
    assert normalize_opcode(" PUSH1   MSTORE\nSTOP ")=="PUSH1 MSTORE STOP"
    assert digest("same")==digest("same")


def test_unknown_is_not_benign_and_predictions_are_ineligible():
    assert normalize_label("unknown") is None
    assert not ground_truth_eligible("D")
    assert ground_truth_eligible("A") and ground_truth_eligible("B")


def test_positive_only_is_not_complete_benchmark_and_thresholds_are_frozen():
    assert not complete_benchmark("A",True,False)
    assert track_b_rating(100,"A",True,True,True)=="STRONG"
    assert track_b_rating(65,"A",False,True,False)=="MODERATE"
    assert track_b_rating(29,"B",True,True,True)=="WEAK"
    assert track_b_rating(537,"B",True,True,False)=="NONE"
    assert track_b_rating(99,"D",True,True,True)=="NONE"


def test_conflicts_and_duplicate_labels_are_preserved():
    internal=add_hashes(pd.DataFrame({"address":["0x"+"1"*40],"label":[0]}))
    external=add_hashes(pd.DataFrame({"address":["0x"+"1"*40]*2,"label":[1,0]}))
    row=address_summary("synthetic",external,internal)
    assert row["external_ponzi_internal_benign"]==1
    assert row["conflicting_internal_duplicate_labels"]==1


def test_download_manifest_has_urls_and_hashes_and_matches_local_files():
    rows=pd.read_csv(OUT/"source_retrieval_manifest.csv",keep_default_na=False)
    downloaded=rows[rows.status=="downloaded"]
    assert len(downloaded)>100
    assert downloaded.canonical_source_url.str.startswith("https://").all()
    assert downloaded.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    for row in downloaded.itertuples():
        path=REPO_ROOT/row.original_filename
        if path.exists(): assert sha256_file(path)==row.sha256


def test_outputs_are_complete_and_no_model_or_p5_evaluation_occurred():
    expected={"phase8r6a_results.json","candidate_sources.csv","source_retrieval_manifest.csv","label_provenance.csv","address_overlap.csv","code_overlap.csv","creator_overlap.csv","temporal_novelty.csv","external_internal_duplicates.csv","cross_source_overlap.csv","candidate_quality_subsets.csv","track_b_feasibility.csv","manual_verification_estimate.csv","external_bias_audit.csv","sealed_set_readiness.csv","phase8r6a_report.md"}
    assert expected=={p.name for p in OUT.iterdir() if p.is_file()}
    r=json.loads((OUT/"phase8r6a_results.json").read_text())
    assert r["models_trained"] is False and r["p5_evaluated"] is False
    assert r["track_b_established"] is False and r["sealed_set_ready"] is False
    assert r["decision"].startswith("B.")
    assert r["next_recommended_phase"]=="external_data_enrichment"


def test_model_predictions_never_marked_primary_ground_truth():
    p=pd.read_csv(OUT/"label_provenance.csv")
    predicted=p[p["class"]=="D"]
    assert not predicted.eligible_primary_track_b.astype(bool).any()
    assert not predicted.prediction_as_ground_truth.astype(bool).any()
