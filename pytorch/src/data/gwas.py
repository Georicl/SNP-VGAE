"""Strict SNP-ID alignment of PLINK association results to BED/BIM order.

Only ADD rows are used when a TEST column is present. Without TEST, each row
is treated as the caller's chosen association test; no genetic model is inferred.
The selected GWAS and BIM SNP sets must match exactly. Missing results, extra
results, duplicates, invalid P values and chromosome/position conflicts are
errors: this module never imputes a P value or silently drops a SNP.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GwasRecord:
    chrom: str
    bp: int
    pvalue: float


def _snp_id(value: str, source: str) -> str:
    if value in {"", ".", "NA"}:
        raise ValueError(f"{source}: missing SNP ID ({value!r})")
    return value


def _position(value: str, source: str) -> int:
    try:
        bp = int(value)
    except ValueError as exc:
        raise ValueError(f"{source}: invalid base-pair position {value!r}") from exc
    if bp <= 0:
        raise ValueError(f"{source}: base-pair position must be positive, got {bp}")
    return bp


def _chromosome(value: str, source: str) -> str:
    # Do not map species-dependent numeric sex-chromosome codes or genome builds.
    if value in {"", ".", "NA", "0"}:
        raise ValueError(f"{source}: missing/unknown chromosome {value!r}")
    return value


def load_bim(path: str | Path) -> list[dict]:
    """Read SNP metadata in BED row order, rejecting ambiguous identifiers."""
    info, seen = [], set()
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            source = f"{path}:{line_no}"
            if len(fields) != 6:
                raise ValueError(f"{source}: BIM requires 6 columns, got {len(fields)}")
            snp = _snp_id(fields[1], source)
            if snp in seen:
                raise ValueError(f"{source}: duplicate BIM SNP ID {snp!r}")
            seen.add(snp)
            info.append({"chr": _chromosome(fields[0], source), "name": snp,
                         "bp": _position(fields[3], source)})
    if not info:
        raise ValueError(f"{path}: BIM contains no SNPs")
    return info


def load_gwas(path: str | Path) -> dict[str, GwasRecord]:
    """Parse columns by header and return unique SNP results (ADD if TEST exists)."""
    records = {}
    with open(path) as handle:
        header = handle.readline().split()
        required = {"CHR", "SNP", "BP", "P"}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"{path}: missing GWAS column(s): {', '.join(missing)}")
        if len(set(header)) != len(header):
            raise ValueError(f"{path}: duplicate GWAS column names")
        columns = {name: index for index, name in enumerate(header)}
        for line_no, line in enumerate(handle, 2):
            if not line.strip():
                continue
            fields = line.split()
            source = f"{path}:{line_no}"
            if len(fields) != len(header):
                raise ValueError(f"{source}: expected {len(header)} GWAS columns, "
                                 f"got {len(fields)}")
            if "TEST" in columns and fields[columns["TEST"]] != "ADD":
                continue
            snp = _snp_id(fields[columns["SNP"]], source)
            if snp in records:
                raise ValueError(f"{source}: duplicate selected GWAS SNP ID {snp!r}; "
                                 "expected one result per SNP after ADD filtering")
            value = fields[columns["P"]]
            try:
                pvalue = float(value)
            except ValueError as exc:
                raise ValueError(f"{source}: SNP {snp!r} has invalid/missing "
                                 f"GWAS P value {value!r}; no imputation is performed") from exc
            if not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
                raise ValueError(f"{source}: SNP {snp!r} has invalid GWAS P value "
                                 f"{value!r}; expected a finite number in [0, 1]")
            records[snp] = GwasRecord(
                chrom=_chromosome(fields[columns["CHR"]], source),
                bp=_position(fields[columns["BP"]], source), pvalue=pvalue)
    if not records:
        qualifier = " ADD" if "TEST" in columns else ""
        raise ValueError(f"{path}: no selected{qualifier} GWAS rows")
    return records


def load_aligned_gwas(
    gwas_path: str | Path,
    bim_path: str | Path,
    *,
    expected_n_snps: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Return BIM metadata and P values in BIM order under a strict 1:1 join.

    Chromosome labels must match exactly, as must positive base-pair positions.
    Harmonize chromosome naming and genome builds explicitly before calling.
    Zero P values are retained as supplied (e.g. numerical underflow), never
    substituted for missing values. ``expected_n_snps`` checks genotype row count.
    """
    bim = load_bim(bim_path)
    if expected_n_snps is not None and len(bim) != expected_n_snps:
        raise ValueError(f"BIM/genotype SNP count mismatch: BIM has {len(bim)} "
                         f"rows but genotype has {expected_n_snps}")
    records = load_gwas(gwas_path)
    bim_ids = {row["name"] for row in bim}
    missing = sorted(bim_ids - records.keys())
    extra = sorted(records.keys() - bim_ids)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing GWAS results for {len(missing)} BIM SNP(s): "
                           + ", ".join(missing[:5]))
        if extra:
            details.append(f"{len(extra)} GWAS SNP(s) absent from BIM: "
                           + ", ".join(extra[:5]))
        raise ValueError("SNP-ID alignment failed (strict unmatched-ID policy): "
                         + "; ".join(details)
                         + ". Reconcile the tested SNP sets explicitly; no rows are "
                           "dropped and no P values are imputed.")
    pvalues = np.empty(len(bim), dtype=np.float64)
    for index, row in enumerate(bim):
        record = records[row["name"]]
        if record.chrom != row["chr"] or record.bp != row["bp"]:
            raise ValueError(f"SNP {row['name']!r}: chromosome/position mismatch: "
                             f"BIM={row['chr']}:{row['bp']}, "
                             f"GWAS={record.chrom}:{record.bp}. Check chromosome "
                             "labels and genome build; no remapping is performed.")
        pvalues[index] = record.pvalue
    return bim, pvalues
