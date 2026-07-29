#!/usr/bin/env python3
"""Construct clinically strict MSK-CHORD stage-IV CRC FOLFOX vs FOLFIRI cohorts.

Primary target trial:
    Population: microsatellite-stable stage-IV colorectal adenocarcinoma,
    first captured systemic treatment line within 120 days of documented
    metastatic diagnosis, no prior outside anticancer medication recorded.
    Treatment: 0 = FOLFOX, 1 = FOLFIRI.
    Outcome: months alive during the first 24 months after treatment initiation.

The model-ready continuous-outcome file excludes patients whose follow-up ended
alive before 24 months because their restricted survival is not directly known.
The full survival file preserves those patients with time/event/censoring fields.
"""
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.44
LINE_GAP_DAYS = 30
MET_DIAGNOSIS_LEAD_DAYS = 30
MAX_DAYS_TO_FIRST_LINE = 120
ECOG_LOOKBACK_DAYS = 90
CEA_LOOKBACK_DAYS = 90
SITE_LOOKBACK_DAYS = 180
MOLECULAR_LOOKBACK_DAYS = 730

SYSTEMIC_SUBTYPES = {"Chemo", "Biologic", "Targeted", "Immuno", "Investigational"}
FOLFOX_REQUIRED = {"FLUOROURACIL", "LEUCOVORIN", "OXALIPLATIN"}
FOLFIRI_REQUIRED = {"FLUOROURACIL", "LEUCOVORIN", "IRINOTECAN"}
ALLOWED_BIOLOGICS = {"BEVACIZUMAB", "CETUXIMAB", "PANITUMUMAB"}
RAS_HOTSPOTS = {12, 13, 59, 61, 117, 146}
CRC_SITE_PATTERN = r"COLON|RECTUM|RECTOSIGMOID|CECUM|HEPATIC FLEXURE|SPLENIC FLEXURE"


def safe_extract(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if target != root and not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tar.extractall(out_dir)
    candidates = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one top-level study directory")
    return candidates[0]


def load_inputs(base: Path) -> dict[str, pd.DataFrame]:
    raw = {
        "patient": pd.read_csv(base / "data_clinical_patient.txt", sep="\t", comment="#", dtype=str),
        "sample": pd.read_csv(base / "data_clinical_sample.txt", sep="\t", comment="#", dtype=str),
        "treatment": pd.read_csv(base / "data_timeline_treatment.txt", sep="\t", dtype=str),
        "diagnosis": pd.read_csv(base / "data_timeline_diagnosis.txt", sep="\t", dtype=str),
        "specimen": pd.read_csv(base / "data_timeline_specimen.txt", sep="\t", dtype=str),
        "progression": pd.read_csv(base / "data_timeline_progression.txt", sep="\t", dtype=str),
        "performance": pd.read_csv(base / "data_timeline_performance_status.txt", sep="\t", dtype=str),
        "tumor_sites": pd.read_csv(base / "data_timeline_tumor_sites.txt", sep="\t", dtype=str),
        "cea": pd.read_csv(base / "data_timeline_cea_labs.txt", sep="\t", dtype=str),
    }
    for key, columns in {
        "treatment": ["START_DATE", "STOP_DATE"],
        "diagnosis": ["START_DATE", "STOP_DATE"],
        "specimen": ["START_DATE", "STOP_DATE"],
        "progression": ["START_DATE", "STOP_DATE"],
        "performance": ["START_DATE", "STOP_DATE", "ECOG"],
        "tumor_sites": ["START_DATE", "STOP_DATE"],
        "cea": ["START_DATE", "STOP_DATE", "RESULT"],
    }.items():
        for column in columns:
            raw[key][column] = pd.to_numeric(raw[key][column], errors="coerce")
    return raw


def classify_side(site: object) -> str:
    value = str(site or "").strip().lower()
    if any(x in value for x in ["cecum", "ascending", "hepatic flexure", "transverse"]):
        return "Right"
    if any(x in value for x in ["splenic flexure", "descending", "sigmoid", "rectosigmoid", "rectum"]):
        return "Left"
    if "colon" in value or "bowel" in value:
        return "Colon NOS"
    return "Unknown"


def histology_group(detailed: object) -> str:
    value = str(detailed or "").lower()
    if "signet" in value:
        return "Signet-ring adenocarcinoma"
    if "mucinous" in value:
        return "Mucinous adenocarcinoma"
    return "Conventional adenocarcinoma"


def eligible_crc_patients(patient: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    stage4 = patient[patient["STAGE_HIGHEST_RECORDED"].eq("Stage 4")][["PATIENT_ID"]]
    crc = sample[sample["CANCER_TYPE"].eq("Colorectal Cancer")].copy()
    crc = crc[
        crc["CANCER_TYPE_DETAILED"].fillna("").str.contains("Adenocarcinoma", case=False)
        & ~crc["CANCER_TYPE_DETAILED"].fillna("").str.contains("In Situ", case=False)
    ]
    crc = crc.sort_values(["PATIENT_ID", "SAMPLE_ID"]).drop_duplicates("PATIENT_ID")
    crc["PRIMARY_SIDE"] = crc["PRIMARY_SITE"].map(classify_side)
    crc["HISTOLOGY_GROUP"] = crc["CANCER_TYPE_DETAILED"].map(histology_group)
    return stage4.merge(crc, on="PATIENT_ID", how="inner")


def diagnosis_dates(diagnosis: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    d = diagnosis[diagnosis["PATIENT_ID"].isin(patient_ids)].copy()
    site_match = d["DX_DESCRIPTION"].fillna("").str.upper().str.contains(CRC_SITE_PATTERN, regex=True)
    d = d[site_match]
    primary = d.groupby("PATIENT_ID")["START_DATE"].min().rename("CRC_PRIMARY_DX_DATE")
    metastatic = (
        d[d["STAGE_CDM_DERIVED"].eq("Stage 4")]
        .groupby("PATIENT_ID")["START_DATE"]
        .min()
        .rename("METASTATIC_DX_DATE")
    )
    result = pd.concat([primary, metastatic], axis=1).reset_index()
    result["DAYS_PRIMARY_TO_METASTATIC"] = result["METASTATIC_DX_DATE"] - result["CRC_PRIMARY_DX_DATE"]
    result["SYNCHRONOUS_METASTATIC"] = result["DAYS_PRIMARY_TO_METASTATIC"].le(90).astype("Int64")
    return result


def assign_treatment_lines(treatment: pd.DataFrame, patient_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = treatment[
        treatment["PATIENT_ID"].isin(patient_ids)
        & treatment["SUBTYPE"].isin(SYSTEMIC_SUBTYPES)
    ].copy()
    t["AGENT"] = t["AGENT"].str.upper()
    t["STOP_EFFECTIVE"] = t["STOP_DATE"].fillna(t["START_DATE"])
    t = t.sort_values(["PATIENT_ID", "START_DATE", "AGENT"]).reset_index(drop=True)
    t["LINE_NUMBER_ALL_SYSTEMIC"] = 0
    for _, group in t.groupby("PATIENT_ID", sort=False):
        line = 1
        previous_start = None
        lines: list[int] = []
        for start in group["START_DATE"]:
            if previous_start is not None and pd.notna(start) and pd.notna(previous_start):
                if abs(start - previous_start) > LINE_GAP_DAYS:
                    line += 1
            lines.append(line)
            previous_start = start
        t.loc[group.index, "LINE_NUMBER_ALL_SYSTEMIC"] = lines

    records: list[dict] = []
    for (pid, line), group in t.groupby(["PATIENT_ID", "LINE_NUMBER_ALL_SYSTEMIC"]):
        agents = set(group["AGENT"].dropna())
        regimen = "OTHER"
        if FOLFOX_REQUIRED | {"IRINOTECAN"} <= agents:
            regimen = "FOLFOXIRI"
        elif FOLFOX_REQUIRED <= agents and "IRINOTECAN" not in agents:
            regimen = "FOLFOX"
        elif FOLFIRI_REQUIRED <= agents and "OXALIPLATIN" not in agents:
            regimen = "FOLFIRI"

        partner = "No biologic"
        if "BEVACIZUMAB" in agents:
            partner = "Bevacizumab"
        if agents & {"CETUXIMAB", "PANITUMUMAB"}:
            partner = "Anti-EGFR" if partner == "No biologic" else "Multiple biologics"

        expected = set()
        if regimen == "FOLFOX":
            expected = FOLFOX_REQUIRED | ALLOWED_BIOLOGICS
        elif regimen == "FOLFIRI":
            expected = FOLFIRI_REQUIRED | ALLOWED_BIOLOGICS
        additional = sorted(agents - expected) if expected else sorted(agents)
        records.append(
            {
                "PATIENT_ID": pid,
                "LINE_NUMBER_ALL_SYSTEMIC": int(line),
                "LINE_START": group["START_DATE"].min(),
                "LINE_STOP": group["STOP_EFFECTIVE"].max(),
                "REGIMEN": regimen,
                "BIOLOGIC_PARTNER": partner,
                "AGENT_LIST": "|".join(sorted(agents)),
                "OTHER_CONCOMITANT_SYSTEMIC": "|".join(additional),
                "HAS_OTHER_CONCOMITANT_SYSTEMIC": bool(additional),
            }
        )
    return t, pd.DataFrame(records)


def choose_first_metastatic_line(lines: pd.DataFrame, diagnosis: pd.DataFrame) -> pd.DataFrame:
    l = lines.merge(diagnosis, on="PATIENT_ID", how="inner")
    l = l[
        (l["LINE_STOP"] >= l["METASTATIC_DX_DATE"] - MET_DIAGNOSIS_LEAD_DAYS)
        & (
            (l["LINE_START"] >= l["METASTATIC_DX_DATE"] - MET_DIAGNOSIS_LEAD_DAYS)
            | (l["LINE_STOP"] >= l["METASTATIC_DX_DATE"])
        )
    ].copy()
    l = l.sort_values(["PATIENT_ID", "LINE_START", "LINE_NUMBER_ALL_SYSTEMIC"])
    first = l.drop_duplicates("PATIENT_ID", keep="first").copy()
    first["DAYS_MET_DX_TO_TREATMENT"] = first["LINE_START"] - first["METASTATIC_DX_DATE"]
    return first


def latest_baseline_value(
    table: pd.DataFrame,
    cohort: pd.DataFrame,
    value_columns: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    records: list[dict] = []
    groups = {pid: group for pid, group in table.groupby("PATIENT_ID")}
    empty = table.iloc[0:0]
    for row in cohort[["PATIENT_ID", "LINE_START"]].itertuples(index=False):
        group = groups.get(row.PATIENT_ID, empty)
        eligible = group[
            group["START_DATE"].le(row.LINE_START)
            & group["START_DATE"].ge(row.LINE_START - lookback_days)
        ]
        record = {"PATIENT_ID": row.PATIENT_ID}
        if not eligible.empty:
            best = eligible.sort_values("START_DATE").iloc[-1]
            for column in value_columns:
                record[column] = best.get(column)
            record["BASELINE_MEASUREMENT_LOOKBACK_DAYS"] = row.LINE_START - best["START_DATE"]
        records.append(record)
    return pd.DataFrame(records)


def baseline_sites(tumor_sites: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    categories = [
        "Liver", "Lung", "Intra-Abdominal", "Lymph Nodes", "Bone",
        "CNS/Brain", "Pleura", "Adrenal Glands", "Reproductive Organs", "Other",
    ]
    records: list[dict] = []
    groups = {pid: group for pid, group in tumor_sites.groupby("PATIENT_ID")}
    empty = tumor_sites.iloc[0:0]
    for row in cohort[["PATIENT_ID", "LINE_START"]].itertuples(index=False):
        group = groups.get(row.PATIENT_ID, empty)
        eligible = group[
            group["START_DATE"].le(row.LINE_START)
            & group["START_DATE"].ge(row.LINE_START - SITE_LOOKBACK_DAYS)
        ]
        observed = set(eligible["TUMOR_SITE"].dropna())
        record = {
            "PATIENT_ID": row.PATIENT_ID,
            "BASELINE_SITE_DATA_AVAILABLE": int(not eligible.empty),
            "BASELINE_METASTATIC_SITE_COUNT": len(observed) if not eligible.empty else np.nan,
        }
        for category in categories:
            column = "SITE_" + category.upper().replace("/", "_").replace("-", "_").replace(" ", "_")
            record[column] = int(category in observed) if not eligible.empty else np.nan
        records.append(record)
    return pd.DataFrame(records)


def baseline_specimen(specimen: pd.DataFrame, sample: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    merged = specimen.merge(sample, on=["SAMPLE_ID", "PATIENT_ID"], how="left")
    merged = merged[merged["CANCER_TYPE"].eq("Colorectal Cancer")]
    records: list[dict] = []
    groups = {pid: group for pid, group in merged.groupby("PATIENT_ID")}
    empty = merged.iloc[0:0]
    keep = [
        "SAMPLE_ID", "START_DATE", "GENE_PANEL", "TMB_NONSYNONYMOUS", "MSI_TYPE",
        "MSI_SCORE", "TUMOR_PURITY", "SAMPLE_TYPE", "METASTATIC_SITE",
    ]
    for row in cohort[["PATIENT_ID", "LINE_START"]].itertuples(index=False):
        group = groups.get(row.PATIENT_ID, empty)
        eligible = group[group["START_DATE"].le(row.LINE_START)]
        record = {"PATIENT_ID": row.PATIENT_ID}
        if not eligible.empty:
            best = eligible.sort_values(["START_DATE", "SAMPLE_ID"]).iloc[-1]
            for column in keep:
                record["BASELINE_" + column] = best.get(column)
            record["BASELINE_SAMPLE_LOOKBACK_DAYS"] = row.LINE_START - best["START_DATE"]
        records.append(record)
    return pd.DataFrame(records)


def mutation_flags(base: Path, sample_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification",
        "HGVSp_Short", "Protein_position",
    ]
    m = pd.read_csv(base / "data_mutations.txt", sep="\t", usecols=usecols, dtype=str, low_memory=False)
    m = m[m["Tumor_Sample_Barcode"].isin(sample_ids) & m["Hugo_Symbol"].isin(["KRAS", "NRAS", "BRAF", "TP53", "PIK3CA"])].copy()
    m["POSITION_INT"] = pd.to_numeric(m["Protein_position"].str.extract(r"(\d+)")[0], errors="coerce")
    excluded = {"Silent", "Intron", "3'UTR", "5'UTR", "RNA", "IGR", "Splice_Region"}
    protein = ~m["Variant_Classification"].isin(excluded)
    m["KRAS_MUT"] = m["Hugo_Symbol"].eq("KRAS") & protein
    m["NRAS_MUT"] = m["Hugo_Symbol"].eq("NRAS") & protein
    m["RAS_HOTSPOT_MUT"] = m["Hugo_Symbol"].isin(["KRAS", "NRAS"]) & protein & m["POSITION_INT"].isin(RAS_HOTSPOTS)
    m["BRAF_V600E_MUT"] = m["Hugo_Symbol"].eq("BRAF") & m["HGVSp_Short"].eq("p.V600E")
    m["TP53_MUT"] = m["Hugo_Symbol"].eq("TP53") & protein
    m["PIK3CA_MUT"] = m["Hugo_Symbol"].eq("PIK3CA") & protein
    if m.empty:
        return pd.DataFrame(columns=["BASELINE_SAMPLE_ID"])
    out = m.groupby("Tumor_Sample_Barcode")[["KRAS_MUT", "NRAS_MUT", "RAS_HOTSPOT_MUT", "BRAF_V600E_MUT", "TP53_MUT", "PIK3CA_MUT"]].max().reset_index()
    return out.rename(columns={"Tumor_Sample_Barcode": "BASELINE_SAMPLE_ID"})


def prior_backbone_exposure(treatment: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    groups = {pid: group for pid, group in treatment.groupby("PATIENT_ID")}
    empty = treatment.iloc[0:0]
    for row in cohort[["PATIENT_ID", "LINE_START", "METASTATIC_DX_DATE"]].itertuples(index=False):
        group = groups.get(row.PATIENT_ID, empty)
        prior = group[group["START_DATE"].lt(row.LINE_START)]
        ox = prior[prior["AGENT"].str.upper().eq("OXALIPLATIN")]["START_DATE"]
        ir = prior[prior["AGENT"].str.upper().eq("IRINOTECAN")]["START_DATE"]
        ox_last = ox.max() if not ox.empty else np.nan
        ir_last = ir.max() if not ir.empty else np.nan
        records.append(
            {
                "PATIENT_ID": row.PATIENT_ID,
                "PRIOR_OXALIPLATIN_ANY": int(not ox.empty),
                "PRIOR_IRINOTECAN_ANY": int(not ir.empty),
                "PRIOR_OXALIPLATIN_LAST_DATE": ox_last,
                "PRIOR_IRINOTECAN_LAST_DATE": ir_last,
                "PRIOR_OXALIPLATIN_WITHIN_12M_OF_MET_DX": int(pd.notna(ox_last) and (row.METASTATIC_DX_DATE - ox_last <= 365)),
                "PRIOR_IRINOTECAN_WITHIN_12M_OF_MET_DX": int(pd.notna(ir_last) and (row.METASTATIC_DX_DATE - ir_last <= 365)),
            }
        )
    return pd.DataFrame(records)


def progression_endpoint(progression: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    groups = {pid: group for pid, group in progression.groupby("PATIENT_ID")}
    empty = progression.iloc[0:0]
    for row in cohort[["PATIENT_ID", "LINE_START"]].itertuples(index=False):
        group = groups.get(row.PATIENT_ID, empty)
        post = group[
            group["START_DATE"].ge(row.LINE_START)
            & group["PROGRESSION"].eq("Y")
        ]
        first = post["START_DATE"].min() if not post.empty else np.nan
        records.append(
            {
                "PATIENT_ID": row.PATIENT_ID,
                "FIRST_RECORDED_PROGRESSION_DATE": first,
                "DAYS_TO_RECORDED_PROGRESSION": first - row.LINE_START if pd.notna(first) else np.nan,
            }
        )
    return pd.DataFrame(records)


def build(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_inputs(base)
    patient, sample = raw["patient"], raw["sample"]
    crc = eligible_crc_patients(patient, sample)
    dx = diagnosis_dates(raw["diagnosis"], set(crc["PATIENT_ID"]))
    _, lines = assign_treatment_lines(raw["treatment"], set(crc["PATIENT_ID"]))
    first = choose_first_metastatic_line(lines, dx)
    cohort = crc.merge(dx, on="PATIENT_ID", how="left").merge(first, on=["PATIENT_ID", "CRC_PRIMARY_DX_DATE", "METASTATIC_DX_DATE", "DAYS_PRIMARY_TO_METASTATIC", "SYNCHRONOUS_METASTATIC"], how="left")
    cohort = cohort.merge(patient, on="PATIENT_ID", how="left", suffixes=("", "_PATIENT"))

    # Baseline variables are computed for patients with an identifiable first metastatic line.
    has_line = cohort[cohort["LINE_START"].notna()].copy()
    prior = prior_backbone_exposure(raw["treatment"], has_line)
    ecog = latest_baseline_value(raw["performance"], has_line, ["ECOG"], ECOG_LOOKBACK_DAYS).rename(columns={"BASELINE_MEASUREMENT_LOOKBACK_DAYS": "ECOG_LOOKBACK_DAYS"})
    cea = latest_baseline_value(raw["cea"], has_line, ["RESULT"], CEA_LOOKBACK_DAYS).rename(columns={"RESULT": "CEA_BASELINE", "BASELINE_MEASUREMENT_LOOKBACK_DAYS": "CEA_LOOKBACK_DAYS"})
    bsites = baseline_sites(raw["tumor_sites"], has_line)
    bspec = baseline_specimen(raw["specimen"], sample, has_line)
    prog = progression_endpoint(raw["progression"], has_line)

    for addon in [prior, ecog, cea, bsites, bspec, prog]:
        cohort = cohort.merge(addon, on="PATIENT_ID", how="left")

    cohort["ECOG"] = pd.to_numeric(cohort["ECOG"], errors="coerce")
    cohort["CEA_BASELINE"] = pd.to_numeric(cohort["CEA_BASELINE"], errors="coerce")
    cohort["CEA_LOG1P"] = np.log1p(cohort["CEA_BASELINE"].clip(lower=0))
    cohort["CURRENT_AGE_DEID"] = pd.to_numeric(cohort["CURRENT_AGE_DEID"], errors="coerce")
    cohort["BASELINE_TMB_NONSYNONYMOUS"] = pd.to_numeric(cohort.get("BASELINE_TMB_NONSYNONYMOUS"), errors="coerce")
    cohort["BASELINE_MSI_SCORE"] = pd.to_numeric(cohort.get("BASELINE_MSI_SCORE"), errors="coerce")
    cohort["BASELINE_TUMOR_PURITY"] = pd.to_numeric(cohort.get("BASELINE_TUMOR_PURITY"), errors="coerce")

    baseline_sample_ids = set(cohort.get("BASELINE_SAMPLE_ID", pd.Series(dtype=str)).dropna())
    if baseline_sample_ids:
        mut = mutation_flags(base, baseline_sample_ids)
        cohort = cohort.merge(mut, on="BASELINE_SAMPLE_ID", how="left")
    for column in ["KRAS_MUT", "NRAS_MUT", "RAS_HOTSPOT_MUT", "BRAF_V600E_MUT", "TP53_MUT", "PIK3CA_MUT"]:
        if column not in cohort:
            cohort[column] = np.nan

    cohort["PRETREATMENT_SAMPLE_AVAILABLE"] = cohort.get("BASELINE_SAMPLE_ID").notna().astype(int)
    cohort["PRETREATMENT_SAMPLE_WITHIN_730D"] = (
        cohort.get("BASELINE_SAMPLE_LOOKBACK_DAYS").le(MOLECULAR_LOOKBACK_DAYS)
    ).fillna(False).astype(int)

    cohort["TREATMENT"] = np.where(cohort["REGIMEN"].eq("FOLFIRI"), 1, np.where(cohort["REGIMEN"].eq("FOLFOX"), 0, np.nan))
    cohort["TREATMENT_LABEL"] = cohort["REGIMEN"]
    cohort["SEX_FEMALE"] = cohort["GENDER"].eq("Female").astype(int)
    cohort["PRIMARY_SIDE_RIGHT"] = np.where(cohort["PRIMARY_SIDE"].eq("Right"), 1, np.where(cohort["PRIMARY_SIDE"].eq("Left"), 0, np.nan))
    cohort["PRIMARY_RECTAL"] = cohort["PRIMARY_SITE"].fillna("").str.lower().str.contains("rect").astype(int)
    cohort["BIOLOGIC_BEVACIZUMAB"] = cohort["BIOLOGIC_PARTNER"].eq("Bevacizumab").astype(int)
    cohort["BIOLOGIC_ANTI_EGFR"] = cohort["BIOLOGIC_PARTNER"].eq("Anti-EGFR").astype(int)
    cohort["ALLOWED_PRIMARY_PARTNER"] = cohort["BIOLOGIC_PARTNER"].isin(["No biologic", "Bevacizumab"])
    cohort["AGE_PROXY_CURRENT_DEID"] = cohort["CURRENT_AGE_DEID"]
    cohort["ECOG_MISSING"] = cohort["ECOG"].isna().astype(int)
    cohort["CEA_MISSING"] = cohort["CEA_BASELINE"].isna().astype(int)

    cohort["OS_MONTHS"] = pd.to_numeric(cohort["OS_MONTHS"], errors="coerce")
    cohort["OS_REFERENCE_DAYS"] = cohort["OS_MONTHS"] * DAYS_PER_MONTH
    cohort["FOLLOWUP_DAYS_FROM_INDEX"] = cohort["OS_REFERENCE_DAYS"] - cohort["LINE_START"]
    cohort["FOLLOWUP_MONTHS_FROM_INDEX"] = cohort["FOLLOWUP_DAYS_FROM_INDEX"] / DAYS_PER_MONTH
    cohort["DEATH_EVENT"] = cohort["OS_STATUS"].eq("1:DECEASED").astype(int)
    horizon_days = 24 * DAYS_PER_MONTH
    valid_followup = cohort["FOLLOWUP_DAYS_FROM_INDEX"].ge(0)
    death_before = valid_followup & cohort["DEATH_EVENT"].eq(1) & cohort["FOLLOWUP_DAYS_FROM_INDEX"].le(horizon_days)
    observed_beyond = valid_followup & cohort["FOLLOWUP_DAYS_FROM_INDEX"].ge(horizon_days)
    cohort["OUTCOME_RMST24_MONTHS"] = np.where(
        death_before,
        cohort["FOLLOWUP_MONTHS_FROM_INDEX"],
        np.where(observed_beyond, 24.0, np.nan),
    )
    cohort["OUTCOME_DEATH_WITHIN_24M"] = np.where(death_before, 1, np.where(observed_beyond, 0, np.nan))
    cohort["SURVIVAL_TIME_24M_MONTHS"] = np.where(
        valid_followup,
        np.minimum(cohort["FOLLOWUP_MONTHS_FROM_INDEX"], 24.0),
        np.nan,
    )
    cohort["SURVIVAL_EVENT_24M"] = death_before.astype(int)
    cohort["CENSORED_BEFORE_24M"] = (valid_followup & cohort["OUTCOME_RMST24_MONTHS"].isna()).astype(int)

    # Eligibility flags and explicit exclusion reasons.
    cohort["HAS_DOCUMENTED_METASTATIC_DX"] = cohort["METASTATIC_DX_DATE"].notna()
    cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"] = cohort["LINE_START"].notna()
    cohort["EXACT_DOUBLEt"] = cohort["REGIMEN"].isin(["FOLFOX", "FOLFIRI"])
    cohort["START_WITHIN_120D"] = cohort["DAYS_MET_DX_TO_TREATMENT"].between(-MET_DIAGNOSIS_LEAD_DAYS, MAX_DAYS_TO_FIRST_LINE, inclusive="both")
    cohort["NO_OTHER_CONCOMITANT"] = ~cohort["HAS_OTHER_CONCOMITANT_SYSTEMIC"].fillna(True)
    cohort["MSI_STABLE"] = cohort["MSI_TYPE"].eq("Stable")
    cohort["NO_PRIOR_OUTSIDE_MEDICATION"] = cohort["PRIOR_MED_TO_MSK"].eq("No prior medications")
    cohort["NO_RECENT_OXALIPLATIN_IRINOTECAN"] = (
        cohort["PRIOR_OXALIPLATIN_WITHIN_12M_OF_MET_DX"].fillna(0).eq(0)
        & cohort["PRIOR_IRINOTECAN_WITHIN_12M_OF_MET_DX"].fillna(0).eq(0)
    )
    cohort["ECOG_NOT_OVER_2"] = cohort["ECOG"].isna() | cohort["ECOG"].le(2)
    cohort["VALID_FOLLOWUP"] = valid_followup

    cohort["ELIGIBLE_PRIMARY_SURVIVAL"] = (
        cohort["HAS_DOCUMENTED_METASTATIC_DX"]
        & cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"]
        & cohort["EXACT_DOUBLEt"]
        & cohort["START_WITHIN_120D"]
        & cohort["NO_OTHER_CONCOMITANT"]
        & cohort["ALLOWED_PRIMARY_PARTNER"]
        & cohort["MSI_STABLE"]
        & cohort["NO_PRIOR_OUTSIDE_MEDICATION"]
        & cohort["NO_RECENT_OXALIPLATIN_IRINOTECAN"]
        & cohort["ECOG_NOT_OVER_2"]
        & cohort["VALID_FOLLOWUP"]
    )
    cohort["ELIGIBLE_PRIMARY_CONTINUOUS"] = cohort["ELIGIBLE_PRIMARY_SURVIVAL"] & cohort["OUTCOME_RMST24_MONTHS"].notna()
    cohort["ELIGIBLE_BROADER_SENSITIVITY"] = (
        cohort["HAS_DOCUMENTED_METASTATIC_DX"]
        & cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"]
        & cohort["EXACT_DOUBLEt"]
        & cohort["START_WITHIN_120D"]
        & cohort["NO_OTHER_CONCOMITANT"]
        & cohort["ALLOWED_PRIMARY_PARTNER"]
        & cohort["MSI_STABLE"]
        & ~cohort["PRIOR_MED_TO_MSK"].eq("Prior medications to MSK")
        & cohort["NO_RECENT_OXALIPLATIN_IRINOTECAN"]
        & cohort["ECOG_NOT_OVER_2"]
        & cohort["VALID_FOLLOWUP"]
        & cohort["OUTCOME_RMST24_MONTHS"].notna()
    )
    cohort["ELIGIBLE_EXPANDED_BENCHMARK"] = (
        cohort["HAS_DOCUMENTED_METASTATIC_DX"]
        & cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"]
        & cohort["EXACT_DOUBLEt"]
        & cohort["NO_OTHER_CONCOMITANT"]
        & cohort["ALLOWED_PRIMARY_PARTNER"]
        & cohort["MSI_STABLE"]
        & cohort["NO_RECENT_OXALIPLATIN_IRINOTECAN"]
        & cohort["ECOG_NOT_OVER_2"]
        & cohort["VALID_FOLLOWUP"]
        & cohort["OUTCOME_RMST24_MONTHS"].notna()
    )
    cohort["ELIGIBLE_BEVACIZUMAB_ONLY"] = (
        cohort["ELIGIBLE_PRIMARY_CONTINUOUS"]
        & cohort["BIOLOGIC_PARTNER"].eq("Bevacizumab")
    )

    cohort["ELIGIBLE_MOLECULAR_SENSITIVITY"] = (
        cohort["ELIGIBLE_PRIMARY_CONTINUOUS"]
        & cohort["PRETREATMENT_SAMPLE_WITHIN_730D"].eq(1)
    )

    reason_columns = [
        ("HAS_DOCUMENTED_METASTATIC_DX", "No documented stage-IV CRC diagnosis date"),
        ("HAS_FIRST_METASTATIC_SYSTEMIC_LINE", "No captured systemic line after metastatic diagnosis"),
        ("EXACT_DOUBLEt", "First metastatic systemic line was not exact FOLFOX or FOLFIRI"),
        ("START_WITHIN_120D", "Treatment did not start within 120 days of metastatic diagnosis"),
        ("NO_OTHER_CONCOMITANT", "Other systemic agent was present in the index line"),
        ("MSI_STABLE", "Tumour was not documented microsatellite-stable"),
        ("NO_PRIOR_OUTSIDE_MEDICATION", "Prior anticancer medication before MSK was recorded or unknown"),
        ("NO_RECENT_OXALIPLATIN_IRINOTECAN", "Recent prior oxaliplatin or irinotecan exposure"),
        ("ECOG_NOT_OVER_2", "Baseline ECOG was greater than 2"),
        ("VALID_FOLLOWUP", "Follow-up from treatment start was invalid"),
    ]
    def reasons(row: pd.Series) -> str:
        values = [label for column, label in reason_columns if not bool(row.get(column, False))]
        if bool(row.get("ELIGIBLE_PRIMARY_SURVIVAL", False)) and pd.isna(row.get("OUTCOME_RMST24_MONTHS")):
            values.append("Alive with less than 24 months of follow-up")
        return " | ".join(values)
    cohort["EXCLUSION_REASONS_PRIMARY"] = cohort.apply(reasons, axis=1)

    return cohort, lines


def select_agent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PATIENT_ID", "TREATMENT", "TREATMENT_LABEL", "OUTCOME_RMST24_MONTHS",
        "AGE_PROXY_CURRENT_DEID", "SEX_FEMALE", "RACE", "ETHNICITY",
        "PRIMARY_SIDE", "PRIMARY_SIDE_RIGHT", "PRIMARY_RECTAL", "HISTOLOGY_GROUP",
        "SYNCHRONOUS_METASTATIC", "DAYS_PRIMARY_TO_METASTATIC", "DAYS_MET_DX_TO_TREATMENT",
        "ECOG", "ECOG_MISSING", "CEA_BASELINE", "CEA_LOG1P", "CEA_MISSING",
        "BIOLOGIC_PARTNER", "BIOLOGIC_BEVACIZUMAB", "BIOLOGIC_ANTI_EGFR",
        "BASELINE_METASTATIC_SITE_COUNT", "BASELINE_SITE_DATA_AVAILABLE",
        "SITE_LIVER", "SITE_LUNG", "SITE_INTRA_ABDOMINAL", "SITE_LYMPH_NODES",
        "SITE_BONE", "SITE_CNS_BRAIN", "SITE_PLEURA", "SITE_ADRENAL_GLANDS",
        "SITE_REPRODUCTIVE_ORGANS", "SITE_OTHER",
        "MSI_TYPE",
        "PRIOR_OXALIPLATIN_ANY", "PRIOR_IRINOTECAN_ANY",
    ]
    # Prefer pretreatment molecular values when available; retain patient-level sample metadata for audit.
    existing = [c for c in columns if c in frame.columns]
    return frame[existing].copy()


def write_outputs(base: Path, out_dir: Path) -> None:
    cohort, lines = build(base)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "msk_crc_folfox_folfiri_audit.csv"
    survival_path = out_dir / "msk_crc_folfox_folfiri_primary_survival.csv"
    model_path = out_dir / "msk_crc_folfox_folfiri_primary_agent_review.csv"
    broader_path = out_dir / "msk_crc_folfox_folfiri_broader_sensitivity.csv"
    molecular_path = out_dir / "msk_crc_folfox_folfiri_molecular_sensitivity.csv"
    benchmark_path = out_dir / "msk_crc_folfox_folfiri_expanded_benchmark.csv"
    bev_path = out_dir / "msk_crc_folfox_folfiri_bevacizumab_only.csv"
    flow_path = out_dir / "msk_crc_folfox_folfiri_cohort_flow.csv"
    summary_path = out_dir / "msk_crc_folfox_folfiri_arm_summary.csv"

    cohort.to_csv(audit_path, index=False)
    primary_survival = cohort[cohort["ELIGIBLE_PRIMARY_SURVIVAL"]].copy()
    primary_survival.to_csv(survival_path, index=False)
    primary_model = cohort[cohort["ELIGIBLE_PRIMARY_CONTINUOUS"]].copy()
    select_agent_columns(primary_model).to_csv(model_path, index=False)
    cohort[cohort["ELIGIBLE_BROADER_SENSITIVITY"]].to_csv(broader_path, index=False)
    cohort[cohort["ELIGIBLE_MOLECULAR_SENSITIVITY"]].to_csv(molecular_path, index=False)
    cohort[cohort["ELIGIBLE_EXPANDED_BENCHMARK"]].to_csv(benchmark_path, index=False)
    cohort[cohort["ELIGIBLE_BEVACIZUMAB_ONLY"]].to_csv(bev_path, index=False)

    flow = [
        ("Stage-IV colorectal adenocarcinoma source population", len(cohort)),
        ("Documented stage-IV CRC diagnosis", int(cohort["HAS_DOCUMENTED_METASTATIC_DX"].sum())),
        ("Captured first metastatic systemic line", int(cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"].sum())),
        ("First line exact FOLFOX/FOLFIRI", int((cohort["HAS_FIRST_METASTATIC_SYSTEMIC_LINE"] & cohort["EXACT_DOUBLEt"]).sum())),
        ("Started within 120 days", int((cohort["EXACT_DOUBLEt"] & cohort["START_WITHIN_120D"]).sum())),
        ("No other concomitant systemic agent", int((cohort["EXACT_DOUBLEt"] & cohort["START_WITHIN_120D"] & cohort["NO_OTHER_CONCOMITANT"]).sum())),
        ("Allowed partner: none or bevacizumab", int((cohort["EXACT_DOUBLEt"] & cohort["START_WITHIN_120D"] & cohort["NO_OTHER_CONCOMITANT"] & cohort["ALLOWED_PRIMARY_PARTNER"]).sum())),
        ("Microsatellite-stable", int((cohort["EXACT_DOUBLEt"] & cohort["START_WITHIN_120D"] & cohort["NO_OTHER_CONCOMITANT"] & cohort["ALLOWED_PRIMARY_PARTNER"] & cohort["MSI_STABLE"]).sum())),
        ("No prior outside anticancer medication", int((cohort["EXACT_DOUBLEt"] & cohort["START_WITHIN_120D"] & cohort["NO_OTHER_CONCOMITANT"] & cohort["MSI_STABLE"] & cohort["NO_PRIOR_OUTSIDE_MEDICATION"]).sum())),
        ("Primary survival cohort", len(primary_survival)),
        ("Primary continuous-outcome cohort", len(primary_model)),
        ("Molecular sensitivity cohort", int(cohort["ELIGIBLE_MOLECULAR_SENSITIVITY"].sum())),
        ("Bevacizumab-only primary subset", int(cohort["ELIGIBLE_BEVACIZUMAB_ONLY"].sum())),
        ("Expanded observational benchmark", int(cohort["ELIGIBLE_EXPANDED_BENCHMARK"].sum())),
    ]
    pd.DataFrame(flow, columns=["STEP", "N"]).to_csv(flow_path, index=False)

    summary_records = []
    for cohort_name, frame in [
        ("Primary survival", primary_survival),
        ("Primary continuous", primary_model),
        ("Broader sensitivity", cohort[cohort["ELIGIBLE_BROADER_SENSITIVITY"]]),
        ("Molecular sensitivity", cohort[cohort["ELIGIBLE_MOLECULAR_SENSITIVITY"]]),
        ("Bevacizumab only", cohort[cohort["ELIGIBLE_BEVACIZUMAB_ONLY"]]),
        ("Expanded benchmark", cohort[cohort["ELIGIBLE_EXPANDED_BENCHMARK"]]),
    ]:
        for regimen, group in frame.groupby("REGIMEN"):
            summary_records.append(
                {
                    "COHORT": cohort_name,
                    "REGIMEN": regimen,
                    "N": len(group),
                    "DEATHS_WITHIN_24M": int(group["OUTCOME_DEATH_WITHIN_24M"].fillna(0).sum()),
                    "OUTCOME_EVALUABLE_N": int(group["OUTCOME_RMST24_MONTHS"].notna().sum()),
                    "MEAN_RMST24_MONTHS": group["OUTCOME_RMST24_MONTHS"].mean(),
                    "MEDIAN_RMST24_MONTHS": group["OUTCOME_RMST24_MONTHS"].median(),
                    "ECOG_AVAILABLE_PCT": 100 * group["ECOG"].notna().mean(),
                    "CEA_AVAILABLE_PCT": 100 * group["CEA_BASELINE"].notna().mean(),
                    "BASELINE_SITE_DATA_PCT": 100 * group["BASELINE_SITE_DATA_AVAILABLE"].fillna(0).mean(),
                    "BEVACIZUMAB_PARTNER_PCT": 100 * group["BIOLOGIC_PARTNER"].eq("Bevacizumab").mean(),
                    "ANTI_EGFR_PARTNER_PCT": 100 * group["BIOLOGIC_PARTNER"].eq("Anti-EGFR").mean(),
                }
            )
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)

    print("Primary survival cohort:", len(primary_survival), primary_survival["REGIMEN"].value_counts().to_dict())
    print("Primary continuous cohort:", len(primary_model), primary_model["REGIMEN"].value_counts().to_dict())
    print("Broader sensitivity:", int(cohort["ELIGIBLE_BROADER_SENSITIVITY"].sum()))
    print("Molecular sensitivity:", int(cohort["ELIGIBLE_MOLECULAR_SENSITIVITY"].sum()))
    print("Bevacizumab only:", int(cohort["ELIGIBLE_BEVACIZUMAB_ONLY"].sum()))
    print("Expanded benchmark:", int(cohort["ELIGIBLE_EXPANDED_BENCHMARK"].sum()))
    print("Files written to", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="MSK-CHORD directory or .tar.gz archive")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.source.is_dir():
        base = args.source
    else:
        base = safe_extract(args.source, args.output_dir / "_msk_crc_extract")
    write_outputs(base, args.output_dir)


if __name__ == "__main__":
    main()