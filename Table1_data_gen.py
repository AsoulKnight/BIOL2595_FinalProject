import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================
# 1. PATHS
# =========================
PROJECT_ROOT = Path(r"C:\1.lwBrown\BIOL 2595\Final Project\pythonProject")

CLINICAL_ROOT = PROJECT_ROOT / "ADNI dataset" / "Clinical_data"
NEURO_ROOT = CLINICAL_ROOT / "Neuropsychological"
DEMOG_ROOT = CLINICAL_ROOT / "Subject_Demographics"

OUT_DIR = PROJECT_ROOT / "table1_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# =========================
# 2. FILE LOCATIONS
# Adjust if needed
# =========================
FILES = {
    "dxsum": CLINICAL_ROOT / "DXSUM_06Apr2026.csv",
    "ptdemog": DEMOG_ROOT / "PTDEMOG_06Apr2026.csv",
    "mmse": NEURO_ROOT / "MMSE_06Apr2026.csv",
    "adas": NEURO_ROOT / "ADAS_06Apr2026.csv",
    "cdr": NEURO_ROOT / "CDR_06Apr2026.csv",
    "faq": NEURO_ROOT / "FAQ_06Apr2026.csv",
    "gdscale": NEURO_ROOT / "GDSCALE_06Apr2026.csv",
    "npiq": NEURO_ROOT / "NPIQ_06Apr2026.csv",
    "adi": DEMOG_ROOT / "ADI_06Apr2026.csv",
    "amas": DEMOG_ROOT / "AMAS_06Apr2026.csv",
    "rurality": DEMOG_ROOT / "RURALITY_06Apr2026.csv",
}


# =========================
# 3. HELPERS
# =========================
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_visit_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single visit column 'VISIT_KEY' using whichever ADNI visit field exists.
    """
    df = df.copy()

    vis_col = find_first_existing(df, ["VISCODE2", "VISCODE", "Timepoint", "PHASE"])
    if vis_col is None:
        df["VISIT_KEY"] = "UNKNOWN"
    else:
        df["VISIT_KEY"] = df[vis_col].astype(str).str.strip().str.lower()

    return df


def normalize_baseline_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep baseline if present. If baseline not present, keep earliest row per RID.
    """
    df = df.copy()
    if "RID" not in df.columns:
        raise ValueError("RID column not found.")

    df = add_visit_key(df)

    baseline_tags = {"bl", "sc", "scmri", "init", "m0"}
    is_baseline = df["VISIT_KEY"].isin(baseline_tags)

    if is_baseline.any():
        base = df[is_baseline].copy()
        base = base.sort_values(["RID"])
        base = base.drop_duplicates(subset=["RID"], keep="first")
        return base

    # fallback if no explicit baseline visit exists
    df = df.sort_values(["RID"])
    df = df.drop_duplicates(subset=["RID"], keep="first")
    return df


def coalesce_columns(df: pd.DataFrame, candidates: list[str], out_name: str) -> pd.DataFrame:
    """
    Combine multiple possible source columns into one output column.
    """
    df = df.copy()
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        df[out_name] = np.nan
        return df

    s = pd.Series([np.nan] * len(df), index=df.index)
    for c in existing:
        s = s.fillna(df[c])
    df[out_name] = s
    return df


def recode_diagnosis(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip().lower()

    if s in {"cn", "normal", "nl", "healthy control"}:
        return "Normal"
    if "mci" in s:
        return "MCI"
    if s in {"ad", "alzheimers disease", "alzheimer's disease", "alzheimer", "alzheimers"}:
        return "AD"
    if "dementia" in s:
        return "AD"

    numeric_map = {
        "1": "Normal",
        "2": "MCI",
        "3": "AD",
    }
    return numeric_map.get(s, np.nan)


def choose_numeric_col(df: pd.DataFrame, preferred_names: list[str]) -> str | None:
    """
    Return the first preferred column found.
    """
    for col in preferred_names:
        if col in df.columns:
            return col
    return None


def summarize_continuous(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:  # allow partial data
        return f"{s.mean():.2f} ± {s.std():.2f}" if len(s) > 0 else ""
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def summarize_categorical(series: pd.Series, positive_value=None) -> str:
    s = series.dropna()
    if len(s) == 0:
        return ""

    if positive_value is not None:
        count = (s == positive_value).sum()
        pct = 100 * count / len(s)
        return f"{count} ({pct:.1f}%)"

    # fallback = most common category
    top = s.astype(str).value_counts(dropna=True)
    if len(top) == 0:
        return ""
    cat = top.index[0]
    count = top.iloc[0]
    pct = 100 * count / len(s)
    return f"{cat}: {count} ({pct:.1f}%)"


def one_hot_yes(series: pd.Series) -> pd.Series:
    """
    Normalize yes/no style fields.
    """
    s = series.astype(str).str.strip().str.lower()
    yes_vals = {"1", "yes", "y", "true", "present"}
    return s.isin(yes_vals).astype(int)


# =========================
# 4. LOAD TABLES
# =========================
tables = {name: standardize_columns(read_csv_safe(path)) for name, path in FILES.items()}

for name, df in tables.items():
    if not df.empty:
        print(f"{name}: {df.shape}")
        print(f"  columns sample: {df.columns[:12].tolist()}")


# =========================
# 5. PREP DIAGNOSIS TABLE
# =========================
dx = tables["dxsum"].copy()
if dx.empty:
    raise FileNotFoundError("DXSUM file is required to build Table 1.")

if "RID" not in dx.columns:
    raise ValueError("DXSUM must contain RID.")

dx = normalize_baseline_visits(dx)

# Try direct text diagnosis first
dx = coalesce_columns(dx, ["DX_bl", "DX", "DIAGNOSIS"], "DX_RAW")
dx["DX_GROUP"] = dx["DX_RAW"].apply(recode_diagnosis)

# If direct labels are incomplete, derive from ADNI indicator columns
if dx["DX_GROUP"].isna().mean() > 0.2:
    dx["DX_GROUP"] = pd.Series([None] * len(dx), dtype="object")

    if "DXNORM" in dx.columns:
        dx.loc[pd.to_numeric(dx["DXNORM"], errors="coerce") == 1, "DX_GROUP"] = "Normal"

    if "DXMCI" in dx.columns:
        dx.loc[pd.to_numeric(dx["DXMCI"], errors="coerce") == 1, "DX_GROUP"] = "MCI"

    if "DXAD" in dx.columns:
        dx.loc[pd.to_numeric(dx["DXAD"], errors="coerce") == 1, "DX_GROUP"] = "AD"

# Debug check
print("\nDX_GROUP counts in DXSUM:")
print(dx["DX_GROUP"].value_counts(dropna=False))

dx = dx[["RID", "DX_GROUP"]].drop_duplicates()


# =========================
# 6. PREP EACH TABLE AT BASELINE
# =========================
def prep_baseline(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["RID"])
    if "RID" not in df.columns:
        return pd.DataFrame(columns=["RID"])

    df = normalize_baseline_visits(df)

    # keep only columns that exist
    keep_existing = [c for c in keep_cols if c in df.columns]

    # ensure RID is included only once
    keep_existing = ["RID"] + [c for c in keep_existing if c != "RID"]

    # remove duplicate column names if any
    df = df.loc[:, ~df.columns.duplicated()]

    return df[keep_existing].copy()


def keep_one_per_rid(df):
    if df.empty or "RID" not in df.columns:
        return pd.DataFrame(columns=["RID"])

    df = df.copy()
    df = df.sort_values("RID")
    df = df.drop_duplicates(subset=["RID"], keep="first")
    return df

# ---- PTDEMOG
pt = tables["ptdemog"].copy()
pt_keep = ["RID", "PTGENDER", "PTEDUCAT", "PTRACCAT", "VISDATE", "PTDOB"]
pt = prep_baseline(pt, pt_keep)

# Convert to datetime
pt["VISDATE"] = pd.to_datetime(pt["VISDATE"], errors="coerce")
pt["PTDOB"] = pd.to_datetime(pt["PTDOB"], errors="coerce")

# Compute age in years
pt["AGE"] = (pt["VISDATE"] - pt["PTDOB"]).dt.days / 365.25

# debug
print("\nAGE non-missing count:", pt["AGE"].notna().sum())
print("AGE preview:")
print(pt["AGE"].describe())

# ---- AMAS
amas = tables["amas"].copy()
amas = keep_one_per_rid(amas)
amas_item_cols = [c for c in amas.columns if re.fullmatch(r"AMAS\d+", c)]

if amas_item_cols:
    amas["AMAS_SCORE"] = (
        amas[amas_item_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )
else:
    amas["AMAS_SCORE"] = np.nan

amas = amas[["RID", "AMAS_SCORE"]]
# ---- MMSE
mmse = tables["mmse"].copy()
mmse = prep_baseline(mmse, mmse.columns.tolist())

mmse_score_col = choose_numeric_col(
    mmse,
    ["MMSCORE", "MMSE", "MMTOTAL", "TOTSCORE"]
)
if mmse_score_col is None:
    mmse["MMSE_SCORE"] = np.nan
else:
    mmse["MMSE_SCORE"] = pd.to_numeric(mmse[mmse_score_col], errors="coerce")
mmse = mmse[["RID", "MMSE_SCORE"]]

# ---- ADAS
adas = tables["adas"].copy()
adas = prep_baseline(adas, adas.columns.tolist())

adas_score_col = choose_numeric_col(
    adas,
    ["TOTAL11", "TOTSCORE", "ADAS_TOTAL", "ADAS11", "Q1SCORE"]
)
if adas_score_col is None:
    adas["ADAS_SCORE"] = np.nan
else:
    adas["ADAS_SCORE"] = pd.to_numeric(adas[adas_score_col], errors="coerce")
adas = adas[["RID", "ADAS_SCORE"]]

# ---- CDR
cdr = tables["cdr"].copy()
cdr = prep_baseline(cdr, cdr.columns.tolist())

cdr_global_col = choose_numeric_col(cdr, ["CDGLOBAL", "CDRSB", "CDRGLOBAL"])
cdr_sum_col = choose_numeric_col(cdr, ["CDRSB", "SUMBOX", "CDRSB"])

cdr["CDR_GLOBAL"] = pd.to_numeric(cdr[cdr_global_col], errors="coerce") if cdr_global_col else np.nan
cdr["CDR_SB"] = pd.to_numeric(cdr[cdr_sum_col], errors="coerce") if cdr_sum_col else np.nan
cdr = cdr[["RID", "CDR_GLOBAL", "CDR_SB"]]

# ---- FAQ
faq = tables["faq"].copy()
faq = prep_baseline(faq, faq.columns.tolist())

faq_col = choose_numeric_col(faq, ["FAQTOTAL", "FAQ", "FAQ_SCORE", "TOTAL"])
faq["FAQ_TOTAL"] = pd.to_numeric(faq[faq_col], errors="coerce") if faq_col else np.nan
faq = faq[["RID", "FAQ_TOTAL"]]

# ---- GDSCALE
gds = tables["gdscale"].copy()
gds = prep_baseline(gds, gds.columns.tolist())

gds_col = choose_numeric_col(gds, ["GDTOTAL", "GDSCORE", "TOTAL"])
gds["GDS_SCORE"] = pd.to_numeric(gds[gds_col], errors="coerce") if gds_col else np.nan
gds = gds[["RID", "GDS_SCORE"]]

# ---- NPIQ
npiq = tables["npiq"].copy()
npiq = prep_baseline(npiq, npiq.columns.tolist())

# Total score already exists
if "NPISCORE" in npiq.columns:
    npiq["NPIQ_TOTAL"] = pd.to_numeric(npiq["NPISCORE"], errors="coerce")
else:
    npiq["NPIQ_TOTAL"] = np.nan

# Item presence columns
# Common convention:
# NPIA = delusions/agitation-type first symptom depending on form
# NPIJ = anxiety
if "NPIA" in npiq.columns:
    npiq["AGITATION_PRESENT"] = pd.to_numeric(npiq["NPIA"], errors="coerce")
else:
    npiq["AGITATION_PRESENT"] = np.nan

if "NPIJ" in npiq.columns:
    npiq["ANXIETY_PRESENT"] = pd.to_numeric(npiq["NPIJ"], errors="coerce")
else:
    npiq["ANXIETY_PRESENT"] = np.nan

npiq = npiq[["RID", "NPIQ_TOTAL", "ANXIETY_PRESENT", "AGITATION_PRESENT"]]

# ---- ADI
adi = tables["adi"].copy()
adi = keep_one_per_rid(adi)

# Prefer national score
if "ADINATIONAL" in adi.columns:
    adi["ADI_SCORE"] = pd.to_numeric(adi["ADINATIONAL"], errors="coerce")
elif "ADISTATE" in adi.columns:
    adi["ADI_SCORE"] = pd.to_numeric(adi["ADISTATE"], errors="coerce")
else:
    adi["ADI_SCORE"] = np.nan

adi = adi[["RID", "ADI_SCORE"]]


# ---- RURALITY
rural = tables["rurality"].copy()
rural = keep_one_per_rid(rural)

rural_col = find_first_existing(rural, ["RUCA_2010", "RUCA", "RUCC_2023", "RUCC"])
if rural_col is None:
    rural["RURALITY_GROUP"] = np.nan
else:
    rural["RURALITY_GROUP"] = rural[rural_col]

rural = rural[["RID", "RURALITY_GROUP"]]


# =========================
# 7. MERGE BASELINE DATASET
# =========================
merged = dx.copy()

for part in [pt, mmse, adas, cdr, faq, gds, npiq, adi, amas, rural]:
    if not part.empty:
        merged = merged.merge(part, on="RID", how="left")
race_map = {
    "1": "American Indian or Alaska Native",
    "2": "Asian",
    "3": "Native Hawaiian or Other Pacific Islander",
    "4": "Black or African American",
    "5": "White",
    "6": "More than one race",
    "7": "Unknown",
}

if "PTRACCAT" in merged.columns:
    merged["PTRACCAT_LABEL"] = merged["PTRACCAT"].astype(str).map(race_map).fillna(merged["PTRACCAT"].astype(str))
else:
    merged["PTRACCAT_LABEL"] = np.nan
print("\nMerged diagnosis counts BEFORE filtering:")
print(merged["DX_GROUP"].value_counts(dropna=False))
print("Merged shape before filtering:", merged.shape)

# keep only main diagnosis groups for Table 1
merged = merged[merged["DX_GROUP"].isin(["Normal", "MCI", "AD"])].copy()

print("\nMerged diagnosis counts AFTER filtering:")
print(merged["DX_GROUP"].value_counts(dropna=False))
print("Merged shape after filtering:", merged.shape)

# keep only main diagnosis groups for Table 1
merged = merged[merged["DX_GROUP"].isin(["Normal", "MCI", "AD"])].copy()

# recode sex
if "PTGENDER" in merged.columns:
    print("\nPTGENDER unique values:")
    print(merged["PTGENDER"].dropna().astype(str).value_counts())

    merged["FEMALE"] = (merged["PTGENDER"] == 2).astype(float)
else:
    merged["FEMALE"] = np.nan

# store missingness summary
merged["ROW_MISSING_PCT"] = merged.isna().mean(axis=1) * 100

baseline_out = OUT_DIR / "adni_baseline_merged.csv"
merged.to_csv(baseline_out, index=False)
print(f"\nSaved merged baseline dataset to:\n{baseline_out}")

# debug
print("\nADI non-missing count:", merged["ADI_SCORE"].notna().sum())
print("\nAMAS non-missing count:", merged["AMAS_SCORE"].notna().sum())
print("\nRurality non-missing count:", merged["RURALITY_GROUP"].notna().sum())
# =========================
# 8. BUILD TABLE 1
# =========================
groups = ["Total", "Normal", "MCI", "AD"]

def get_subset(df, grp):
    if grp == "Total":
        return df
    return df[df["DX_GROUP"] == grp]


rows = []

def add_continuous_row(label, col):
    row = {"Variable": label}
    for grp in groups:
        sub = get_subset(merged, grp)
        row[grp] = summarize_continuous(sub[col]) if col in sub.columns else ""
    rows.append(row)

def add_categorical_yes_row(label, col, positive=1):
    row = {"Variable": label}
    for grp in groups:
        sub = get_subset(merged, grp)
        row[grp] = summarize_categorical(sub[col], positive_value=positive) if col in sub.columns else ""
    rows.append(row)

def add_top_category_row(label, col):
    row = {"Variable": label}
    for grp in groups:
        sub = get_subset(merged, grp)
        row[grp] = summarize_categorical(sub[col]) if col in sub.columns else ""
    rows.append(row)

def add_n_row():
    row = {"Variable": "N"}
    for grp in groups:
        sub = get_subset(merged, grp)
        row[grp] = len(sub)
    rows.append(row)


add_n_row()

# demographics
add_continuous_row("Age, years", "AGE")
add_categorical_yes_row("Female, n (%)", "FEMALE", positive=1)
add_continuous_row("Education, years", "PTEDUCAT")
add_top_category_row("Most common race category", "PTRACCAT_LABEL")

# cognitive
add_continuous_row("MMSE score", "MMSE_SCORE")
add_continuous_row("ADAS-Cog score", "ADAS_SCORE")
add_continuous_row("CDR Global", "CDR_GLOBAL")
add_continuous_row("CDR Sum of Boxes", "CDR_SB")

# functional / psychiatric
add_continuous_row("FAQ total", "FAQ_TOTAL")
add_continuous_row("GDS score", "GDS_SCORE")
add_continuous_row("NPIQ total", "NPIQ_TOTAL")
add_categorical_yes_row("Anxiety present, n (%)", "ANXIETY_PRESENT", positive=1)
add_categorical_yes_row("Agitation present, n (%)", "AGITATION_PRESENT", positive=1)

# socioeconomic / environment
add_continuous_row("ADI score", "ADI_SCORE")
add_continuous_row("AMAS score", "AMAS_SCORE")
add_top_category_row("Most common rurality category", "RURALITY_GROUP")

# missingness
add_continuous_row("Row missingness, %", "ROW_MISSING_PCT")

table1 = pd.DataFrame(rows)

csv_out = OUT_DIR / "Table1_summary.csv"
xlsx_out = OUT_DIR / "Table1_summary.xlsx"

table1.to_csv(csv_out, index=False)

with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
    table1.to_excel(writer, sheet_name="Table1", index=False)
    merged.to_excel(writer, sheet_name="Merged_Baseline_Data", index=False)

print(f"Saved Table 1 CSV to:\n{csv_out}")
print(f"Saved Table 1 Excel to:\n{xlsx_out}")


# =========================
# 9. OPTIONAL: PRINT QUICK PREVIEW
# =========================
print("\nPreview of Table 1:")
print(table1.head(20).to_string(index=False))