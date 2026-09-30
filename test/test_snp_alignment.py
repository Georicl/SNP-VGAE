"""Synthetic regression tests; no genotype files or model training are required."""

import numpy as np
import pytest

from data.gwas import load_aligned_gwas, load_bim, load_gwas


@pytest.fixture
def bim_path(tmp_path):
    path = tmp_path / "input.bim"
    path.write_text("1 rsA 0 101 A G\n2 rsB 0 202 C T\n")
    return path


def write_gwas(tmp_path, rows, header="CHR SNP BP A1 TEST NMISS BETA STAT P"):
    path = tmp_path / "input.assoc.linear"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_reordered_gwas_is_aligned_to_bim_order(tmp_path, bim_path):
    path = write_gwas(tmp_path, [
        "2 rsB 202 C ADD 20 0.2 2 0.8",
        "1 rsA 101 A ADD 20 0.1 1 0.003",
    ])
    bim, pvalues = load_aligned_gwas(path, bim_path, expected_n_snps=2)
    assert [row["name"] for row in bim] == ["rsA", "rsB"]
    np.testing.assert_array_equal(pvalues, [0.003, 0.8])


def test_header_order_is_not_used_as_a_positional_schema(tmp_path, bim_path):
    path = write_gwas(tmp_path, ["0.8 rsB 202 ADD 2", "0.003 rsA 101 ADD 1"],
                      header="P SNP BP TEST CHR")
    _, pvalues = load_aligned_gwas(path, bim_path)
    np.testing.assert_array_equal(pvalues, [0.003, 0.8])


def test_add_filter_ignores_other_tests_and_their_p_values(tmp_path, bim_path):
    path = write_gwas(tmp_path, [
        "1 rsA 101 A AGE 20 0.4 0 NA",
        "2 rsB 202 C DOMDEV 20 0.3 0 0.00001",
        "2 rsB 202 C ADD 20 0.2 2 0.8",
        "1 rsA 101 A ADD 20 0.1 1 0.003",
    ])
    _, pvalues = load_aligned_gwas(path, bim_path)
    np.testing.assert_array_equal(pvalues, [0.003, 0.8])


def test_missing_snp_fails_instead_of_imputing_or_shifting(tmp_path, bim_path):
    path = write_gwas(tmp_path, ["2 rsB 202 C ADD 20 0.2 2 0.8"])
    with pytest.raises(ValueError, match="missing GWAS results.*rsA"):
        load_aligned_gwas(path, bim_path)


def test_extra_gwas_snp_fails_strict_unmatched_policy(tmp_path, bim_path):
    path = write_gwas(tmp_path, [
        "1 rsA 101 A ADD 20 0.1 1 0.003",
        "2 rsB 202 C ADD 20 0.2 2 0.8",
        "3 rsC 303 G ADD 20 0.1 1 0.05",
    ])
    with pytest.raises(ValueError, match="GWAS SNP.*absent from BIM: rsC"):
        load_aligned_gwas(path, bim_path)


def test_duplicate_selected_gwas_snp_fails_even_if_values_agree(tmp_path, bim_path):
    path = write_gwas(tmp_path, [
        "1 rsA 101 A ADD 20 0.1 1 0.003",
        "1 rsA 101 A ADD 20 0.1 1 0.003",
        "2 rsB 202 C ADD 20 0.2 2 0.8",
    ])
    with pytest.raises(ValueError, match="duplicate selected GWAS SNP ID 'rsA'"):
        load_aligned_gwas(path, bim_path)


def test_duplicate_bim_snp_fails(tmp_path, bim_path):
    bim_path.write_text("1 rsA 0 101 A G\n1 rsA 0 101 A G\n")
    with pytest.raises(ValueError, match="duplicate BIM SNP ID 'rsA'"):
        load_bim(bim_path)


@pytest.mark.parametrize("chrom,bp", [("3", "101"), ("1", "102"), ("chr1", "101")])
def test_same_snp_id_with_coordinate_conflict_fails(tmp_path, bim_path, chrom, bp):
    path = write_gwas(tmp_path, [
        f"{chrom} rsA {bp} A ADD 20 0.1 1 0.003",
        "2 rsB 202 C ADD 20 0.2 2 0.8",
    ])
    with pytest.raises(ValueError, match="rsA.*chromosome/position mismatch"):
        load_aligned_gwas(path, bim_path)


@pytest.mark.parametrize("pvalue", ["NA", "nan", "inf", "-0.1", "1.01"])
def test_invalid_selected_p_value_is_not_fabricated(tmp_path, pvalue):
    path = write_gwas(tmp_path, [f"1 rsA 101 A ADD 20 0.1 1 {pvalue}"])
    with pytest.raises(ValueError, match="rsA.*invalid.*GWAS P value"):
        load_gwas(path)


def test_no_test_column_uses_supplied_results(tmp_path, bim_path):
    path = write_gwas(tmp_path, ["2 rsB 202 1", "1 rsA 101 0"], header="CHR SNP BP P")
    _, pvalues = load_aligned_gwas(path, bim_path)
    np.testing.assert_array_equal(pvalues, [0, 1])


def test_no_add_rows_fails(tmp_path):
    path = write_gwas(tmp_path, ["1 rsA 101 A DOMDEV 20 0.1 1 0.003"])
    with pytest.raises(ValueError, match="no selected ADD GWAS rows"):
        load_gwas(path)


def test_bim_count_must_match_genotype_rows(tmp_path, bim_path):
    path = write_gwas(tmp_path, ["1 rsA 101 0.003", "2 rsB 202 0.8"],
                      header="CHR SNP BP P")
    with pytest.raises(ValueError, match="BIM/genotype SNP count mismatch"):
        load_aligned_gwas(path, bim_path, expected_n_snps=3)


@pytest.mark.parametrize("header", ["CHR SNP BP", "CHR SNP BP P P"])
def test_missing_or_duplicate_header_columns_fail(tmp_path, header):
    path = write_gwas(tmp_path, [], header=header)
    with pytest.raises(ValueError, match="GWAS column"):
        load_gwas(path)


def test_malformed_gwas_row_is_not_silently_skipped(tmp_path):
    path = write_gwas(tmp_path, ["1 rsA 101"], header="CHR SNP BP P")
    with pytest.raises(ValueError, match="expected 4 GWAS columns, got 3"):
        load_gwas(path)


@pytest.mark.parametrize("row", ["1 . 0 101 A G", "1 rsA 0 0 A G", "0 rsA 0 101 A G"])
def test_missing_bim_identity_or_coordinate_fails(bim_path, row):
    bim_path.write_text(row + "\n")
    with pytest.raises(ValueError):
        load_bim(bim_path)


@pytest.mark.parametrize("scores,expected_category", [
    ([1.0, 0.1], "StableCandidate+NominalGWAS"),
    ([0.1, 1.0], "StableCandidate_NoNominalOverlap"),
])
def test_candidate_export_uses_aligned_p_values_and_preserves_legacy_results(
    tmp_path, bim_path, monkeypatch, scores, expected_category,
):
    """Exercise the output path with synthetic scores, without training a model."""
    import csv
    import json
    from types import SimpleNamespace

    import torch
    from test import snp_significance as script

    path = write_gwas(tmp_path, [
        "2 rsB 202 C ADD 20 0.2 2 0.8",
        "1 rsA 101 A ADD 20 0.1 1 0.003",
    ])
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    legacy = out_dir / "snp_significance.tsv"
    legacy.write_text("existing results must be preserved\n")
    monkeypatch.setattr(script, "BIM", str(bim_path))
    monkeypatch.setattr(script, "GWAS", str(path))
    monkeypatch.setattr(script, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(script, "FIG_DIR", str(out_dir / "figures"))
    monkeypatch.setattr(script, "genotype_read", lambda *args: np.zeros((2, 10)))
    monkeypatch.setattr(script, "grm_reader", lambda *args: np.eye(10))
    monkeypatch.setattr(script, "grm_id_reader", lambda *args: [])
    monkeypatch.setattr(script, "load_pheno", lambda *args: np.arange(10, dtype=np.float32))
    monkeypatch.setattr(script, "train_snp_vae", lambda **kwargs: (None, np.zeros((2, 64))))
    monkeypatch.setattr(script, "GraphBuilder", lambda: SimpleNamespace(
        knn_graph_builder=lambda *args, **kwargs: (None, None),
        build_node_features=lambda *args: None,
    ))
    monkeypatch.setattr(script, "VGAEModel", lambda **kwargs: None)
    monkeypatch.setattr(script, "VGAETrainer", lambda **kwargs: SimpleNamespace(train=lambda: None))
    monkeypatch.setattr(script, "generate_kfold_masks", lambda *args: [(None, None, None)] * 5)
    monkeypatch.setattr(script, "compute_snp_scores", lambda **kwargs: torch.tensor(scores))
    script.main()

    with (out_dir / "snp_candidates.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["SNP"] for row in rows] == ["rsA", "rsB"]
    assert [float(row["GWAS_P"]) for row in rows] == [0.003, 0.8]
    with (out_dir / "stable_snp_candidates.tsv").open() as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    assert len(candidates) == 1
    assert candidates[0]["Category"] == expected_category
    summary = json.loads((out_dir / "snp_candidates_summary.json").read_text())
    assert summary["alignment"]["unmatched_policy"] == "error"
    assert summary["counts"]["stable_high_attribution_candidates"] == 1
    assert "not calibrated statistical significance" in summary["interpretation"]["attribution"]
    assert legacy.read_text() == "existing results must be preserved\n"
    assert not (out_dir / "significant_snps.tsv").exists()
    assert len(list((out_dir / "figures").glob("snp_candidate_*.png"))) == 3
