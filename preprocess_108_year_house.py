# -*- coding: utf-8 -*-
"""
Preprocess 108_year_house.csv according to the thesis workflow:
1. Keep only house-building transactions.
2. Remove outliers with boxplot/IQR rules by district.
3. Fill missing values: numeric columns use mean, categorical columns use mode.
4. Normalize numeric feature columns to 0-1 with Min-Max normalization.
5. Convert categorical feature columns to dummy variables.

Usage:
    python preprocess_108_year_house.py

Or:
    python preprocess_108_year_house.py --input "C:/path/108_year_house.csv" --output "C:/path/clean.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"C:\Users\user\Desktop\Housing_Predict\108_year_house.csv")
DEFAULT_OUTPUT = Path("108_year_house_clean_model.csv")

DISTRICT_COL = "鄉鎮市區"
OBJECT_COL = "交易標的"
TARGET_COL = "單價-每平方公尺.Element:Text"

# Columns corresponding to the paper's dropped attributes, plus dataset metadata
# that is constant or not useful for a district-level model.
DROP_COLUMNS = [
    "編號",
    "機關代碼",
    "縣市別代碼",
    "行政區域代碼",
    "土地區段位置-建物區段門牌",
    "非都市土地使用分區",
    "非都市土地使用編定",
    "移轉層次.Element:Text",
    "總層數.Element:Text",
    "主要建材.Element:Text",
    "總價-元",
    "車位類別.Element:Text",
    "備註.Element:Text",
    "地所狀態",
]

NUMERIC_COLUMNS = [
    "土地移轉總面積-平方公尺",
    "民國年月",
    "建築完成年月.Element:Text",
    "屋齡",
    "建物移轉總面積-平方公尺",
    "建物現況格局-房",
    "建物現況格局-廳",
    "建物現況格局-衛",
    "總價-元",
    TARGET_COL,
    "車位移轉總面積-平方公尺",
    "車位總價-元",
]

# Use continuous price/area variables for IQR removal. Doing this by district
# follows the thesis idea that each district is processed independently.
OUTLIER_COLUMNS = [
    TARGET_COL,
    "土地移轉總面積-平方公尺",
    "建物移轉總面積-平方公尺",
    "車位移轉總面積-平方公尺",
    "車位總價-元",
    "屋齡",
]

MISSING_MARKERS = [
    "",
    " ",
    "-",
    "--",
    "NA",
    "N/A",
    "NaN",
    "nan",
    "None",
    "null",
    "[Table]",
]


def read_house_csv(path: Path) -> pd.DataFrame:
    """Read the government CSV. The file is Big5/CP950-like."""
    return pd.read_csv(
        path,
        encoding="cp950",
        encoding_errors="replace",
        low_memory=False,
    )


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    df = df.replace(MISSING_MARKERS, np.nan)
    return df


def to_number(series: pd.Series) -> pd.Series:
    """Convert numeric-looking text to numbers."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_number(df[col])

    return df


def keep_house_building_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Land + Building and Land + Building + Parking Space records."""
    allowed = ["房地(土地+建物)", "房地(土地+建物)+車位"]
    return df[df[OBJECT_COL].isin(allowed)].copy()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Gregorian year/month features from ROC year-month fields."""
    df = df.copy()

    if "民國年月" in df.columns:
        ym = to_number(df["民國年月"])
        df["交易西元年"] = (ym // 100 + 1911).astype("float")
        df["交易月份"] = (ym % 100).astype("float")

    build_col = "建築完成年月.Element:Text"
    if build_col in df.columns:
        build_ym = to_number(df[build_col])
        df["建築完成西元年"] = (build_ym // 100 + 1911).astype("float")
        df["建築完成月份"] = (build_ym % 100).astype("float")

    return df


def remove_invalid_target(df: pd.DataFrame) -> pd.DataFrame:
    """The dependent variable is unit price, so it must exist and be positive."""
    return df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()


def remove_outliers_iqr_by_district(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep = pd.Series(True, index=df.index)

    for _, group in df.groupby(DISTRICT_COL, dropna=False):
        group_keep = pd.Series(True, index=group.index)

        for col in OUTLIER_COLUMNS:
            if col not in group.columns:
                continue

            values = group[col].dropna()
            if values.nunique() < 2:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            group_keep &= group[col].between(lower, upper) | group[col].isna()

        keep.loc[group.index] = group_keep

    return df[keep].copy()


def fill_missing_by_district(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_cols = [col for col in feature_cols if col in df.columns and col not in numeric_cols]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float")
        district_mean = df.groupby(DISTRICT_COL, dropna=False)[col].transform("mean")
        global_mean = df[col].mean()
        fill_value = 0 if pd.isna(global_mean) else global_mean
        df[col] = df[col].fillna(district_mean).fillna(fill_value)

    for col in categorical_cols:
        district_mode = df.groupby(DISTRICT_COL, dropna=False)[col].transform(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else np.nan
        )
        global_mode = df[col].mode(dropna=True)
        fill_value = global_mode.iloc[0] if not global_mode.empty else "未知"
        df[col] = df[col].fillna(district_mode).fillna(fill_value)

    return df


def minmax_normalize_by_district(df: pd.DataFrame, numeric_feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in numeric_feature_cols:
        if col not in df.columns:
            continue

        def normalize_group(s: pd.Series) -> pd.Series:
            min_value = s.min()
            max_value = s.max()
            denominator = max_value - min_value
            if pd.isna(denominator) or denominator == 0:
                return pd.Series(0.0, index=s.index)
            return (s - min_value) / denominator

        df[col] = df.groupby(DISTRICT_COL, dropna=False)[col].transform(normalize_group)

    return df


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned model table with district, target, normalized numeric features, and dummies."""
    df = normalize_missing_values(df)
    df = convert_numeric_columns(df)
    df = keep_house_building_transactions(df)
    df = add_date_features(df)
    df = remove_invalid_target(df)
    df = remove_outliers_iqr_by_district(df)

    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    model_df = df.drop(columns=drop_cols)

    # District is used for per-district preprocessing and kept as metadata,
    # but it is not converted into a model feature.
    feature_cols = [
        col for col in model_df.columns if col not in [DISTRICT_COL, TARGET_COL]
    ]

    model_df = fill_missing_by_district(model_df, feature_cols)

    numeric_feature_cols = [
        col
        for col in feature_cols
        if col in model_df.columns and pd.api.types.is_numeric_dtype(model_df[col])
    ]
    categorical_feature_cols = [
        col for col in feature_cols if col in model_df.columns and col not in numeric_feature_cols
    ]

    model_df = minmax_normalize_by_district(model_df, numeric_feature_cols)

    dummies = pd.get_dummies(
        model_df[categorical_feature_cols],
        prefix=categorical_feature_cols,
        dummy_na=False,
        dtype=int,
    )

    result = pd.concat(
        [
            model_df[[DISTRICT_COL, TARGET_COL]].reset_index(drop=True),
            model_df[numeric_feature_cols].reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return result


def save_district_files(model_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for district, group in model_df.groupby(DISTRICT_COL, dropna=False):
        safe_name = str(district).replace("/", "_").replace("\\", "_")
        group.to_csv(output_dir / f"{safe_name}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess 108_year_house.csv for modeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Cleaned CSV output path.")
    parser.add_argument(
        "--district-output-dir",
        type=Path,
        default=None,
        help="Optional directory for one cleaned CSV per district.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = read_house_csv(args.input)
    clean_df = build_model_table(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.district_output_dir is not None:
        save_district_files(clean_df, args.district_output_dir)

    print(f"Raw rows: {len(raw_df):,}")
    print(f"Clean model rows: {len(clean_df):,}")
    print(f"Clean model columns: {len(clean_df.columns):,}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
