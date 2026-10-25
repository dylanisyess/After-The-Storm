"""Step 1: filter statewide Safe to Swim bacteria data to LA-region samples.

Keeps enterococcus (ocean) and E. coli (freshwater) culture results
(MPN or CFU per 100 mL) from Regional Board 4 (Los Angeles), 2010 onward.
Also keeps fecal and total coliform: the model doesn't use them, but beach
signs are posted when ANY of the three ocean standards fails, so 05 needs them
to score today's signs fairly.
Drops rows the Water Boards flagged as needing extensive review or rejected.
Output: processed/bacteria_la.csv, one row per sample result.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = ["safetoswim_geomeans_2010-2020.csv", "safetoswim_geomeans_2020-present.csv"]
KEEP = ["DataSource", "Program", "StationCode", "StationName", "SampleDate", "SampleDateTime",
        "Analyte", "Unit", "Result", "ResultSub", "ResultQualCode", "TargetLatitude",
        "TargetLongitude", "RegionNumber", "DataQuality", "MatrixName", "SampleTypeCode",
        "CollectionReplicate", "ResultsReplicate", "SampleAgency"]
BAD_QUALITY = {"Extensive review needed", "Reject record", "Reject Data", "Metadata, QC record"}

parts = []
for name in FILES:
    for ch in pd.read_csv(ROOT / name, usecols=KEEP, dtype=str, chunksize=400_000, low_memory=False):
        m = (ch["RegionNumber"] == "4") \
            & ch["Analyte"].isin(["Enterococcus", "E. coli", "Coliform, Fecal", "Coliform, Total"]) \
            & ch["Unit"].isin(["MPN/100 mL", "CFU/100 mL"]) \
            & (ch["MatrixName"] == "samplewater") \
            & ~ch["DataQuality"].isin(BAD_QUALITY)
        parts.append(ch[m])
    print("read", name, file=sys.stderr)

df = pd.concat(parts, ignore_index=True)
# "mixed": most dates are "2023-01-10" but ~2% are "2023-01-10 08:58:00". Without it,
# pandas guesses one format from the first row and silently blanks the others.
df["SampleDate"] = pd.to_datetime(df["SampleDate"], errors="coerce", format="mixed").dt.normalize()
df = df[df["SampleDate"] >= "2010-01-01"]
for c in ["Result", "ResultSub", "TargetLatitude", "TargetLongitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["value"] = df["ResultSub"].fillna(df["Result"])
df = df.dropna(subset=["value", "TargetLatitude", "TargetLongitude"])
# Field/lab replicates of the same sample: keep the original (replicate 1).
df = df[(df["CollectionReplicate"].fillna("1") == "1") & (df["ResultsReplicate"].fillna("1") == "1")]
# Same station, date and analyte reported twice (e.g., the 2020 file overlap): keep one.
df = df.sort_values("SampleDateTime").drop_duplicates(["StationCode", "SampleDate", "Analyte"], keep="first")

df.to_csv(ROOT / "processed" / "bacteria_la.csv", index=False)
print(f"{len(df):,} samples, {df.StationCode.nunique()} stations, "
      f"{df.SampleDate.min().date()} to {df.SampleDate.max().date()}")
print(df.groupby("Analyte").size().to_string())
