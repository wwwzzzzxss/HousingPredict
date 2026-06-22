import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. Read data
# =========================
file_path = "Merged_Housing_Data_With_Coordinates.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

print("Data shape:", df.shape)
print(df.head())
print(df.info())

df = df[
    (df["主要用途_住家用"] == 1)
].copy()


def remove_iqr_outliers_by_district(data):
    district_cols = [
        col for col in data.columns
        if col.startswith("鄉鎮市區_")
    ]
    value_cols = [
        col for col in ["總價元", "單價元平方公尺"]
        if col in data.columns
    ]

    if not district_cols:
        raise ValueError("找不到鄉鎮市區 one-hot 欄位，請確認欄位名稱是否以 '鄉鎮市區_' 開頭。")

    if not value_cols:
        raise ValueError("找不到可清理的價格欄位：總價元、單價元平方公尺。")

    outlier_mask = pd.Series(False, index=data.index)

    for district_col in district_cols:
        district_mask = data[district_col].eq(1)

        if not district_mask.any():
            continue

        district_outlier_mask = pd.Series(False, index=data.index)

        for value_col in value_cols:
            values = data.loc[district_mask, value_col].dropna()

            if values.empty:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            district_outlier_mask |= (
                district_mask
                & data[value_col].notna()
                & (
                    (data[value_col] < lower_bound)
                    | (data[value_col] > upper_bound)
                )
            )

        removed_count = int(district_outlier_mask.sum())

        if removed_count > 0:
            print(f"{district_col} removed outliers: {removed_count}")

        outlier_mask |= district_outlier_mask

    before_count = len(data)
    cleaned_data = data.loc[~outlier_mask].copy()
    after_count = len(cleaned_data)

    print("Rows before outlier removal:", before_count)
    print("Rows after outlier removal:", after_count)
    print("Removed rows:", before_count - after_count)

    return cleaned_data


df = remove_iqr_outliers_by_district(df)


# =========================
# 2. Sort by transaction time
# =========================
df = df.sort_values(["交易年", "交易月"]).reset_index(drop=True)

print("Rows by year:")
print(df["交易年"].value_counts().sort_index())


# =========================
# 3. Time split
# =========================
train_df = df[df["交易年"] <= 2024].copy()
test_df = df[df["交易年"] == 2025].copy()

print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("Train year range:")
print(train_df["交易年"].min(), "~", train_df["交易年"].max())

print("Test year range:")
print(test_df["交易年"].min(), "~", test_df["交易年"].max())


# =========================
# 4. Set target
# =========================
target_col = "總價元"

# Avoid leakage. These columns directly contain or reveal the target price.
leakage_cols = [
    "總價元",
    "單價元平方公尺",
    "真實單價(元/平方公尺)",
]

drop_cols = [col for col in leakage_cols if col in df.columns]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target_col]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target_col]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# =========================
# 5. Preprocessing
# =========================
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# =========================
# 6. Linear regression model
# =========================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])


# =========================
# 7. Train
# =========================
model.fit(X_train, y_train)


# =========================
# 8. Predict and evaluate
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

mean_total_price = y_test.mean()
mae_ratio = mae / mean_total_price

print("Total price MAE:", mae)
print("Total price RMSE:", rmse)
print("Total price R2:", r2)
print("Mean true total price:", mean_total_price)
print("MAE / mean total price:", mae_ratio)
print("MAE / mean total price (%):", mae_ratio * 100, "%")


# =========================
# 9. Show top 20 coefficients
# =========================
preprocessor_fitted = model.named_steps["preprocessor"]
linear_model = model.named_steps["model"]

feature_names = preprocessor_fitted.get_feature_names_out()
coefficients = linear_model.coef_

coef_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients),
}).sort_values(by="abs_coefficient", ascending=False)

print(coef_importance.head(20))


# =========================
# 10. Save model
# =========================
joblib.dump(model, "linear_regression_housing_model_total_price_time_split.pkl")

print("Model saved as linear_regression_housing_model_total_price_time_split.pkl")
