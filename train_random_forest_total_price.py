import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. 讀取資料
# =========================
file_path = "Merged_Housing_Data_With_Coordinates.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

print("資料筆數與欄位數：", df.shape)
print(df.head())
print(df.info())

df = df[
    (df["主要用途_住家用"] == 1)
].copy()

# =========================
# 2. 依時間排序
# =========================
df = df.sort_values(["交易年", "交易月"]).reset_index(drop=True)

print("各年份資料筆數：")
print(df["交易年"].value_counts().sort_index())


# =========================
# 3. 用時間切分訓練集與測試集
# =========================
train_df = df[df["交易年"] <= 2024].copy()
test_df = df[df["交易年"] == 2025].copy()

print("訓練集：", train_df.shape)
print("測試集：", test_df.shape)

print("訓練集年份範圍：")
print(train_df["交易年"].min(), "~", train_df["交易年"].max())

print("測試集年份範圍：")
print(test_df["交易年"].min(), "~", test_df["交易年"].max())


# =========================
# 4. 設定預測目標 y
# =========================
target_col = "總價元"

# 避免資料洩漏：預測總價時，總價本身與可直接推回總價/單價的欄位都不能放進 X。
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

print("X_train：", X_train.shape)
print("X_test：", X_test.shape)


# =========================
# 5. 建立前處理流程
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
# 6. 建立隨機森林模型
# =========================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model),
])


# =========================
# 7. 模型訓練
# =========================
model.fit(X_train, y_train)


# =========================
# 8. 模型預測與評估
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

mean_total_price = y_test.mean()
mae_ratio = mae / mean_total_price

print("總價 MAE：", mae)
print("總價 RMSE：", rmse)
print("2025 測試集總價 R2：", r2)
print("平均真實總價：", mean_total_price)
print("MAE 佔平均總價比例：", mae_ratio)
print("MAE 佔平均總價比例百分比：", mae_ratio * 100, "%")


# =========================
# 9. 顯示前 20 個重要特徵
# =========================
preprocessor_fitted = model.named_steps["preprocessor"]
rf_fitted = model.named_steps["model"]

feature_names = preprocessor_fitted.get_feature_names_out()
importances = rf_fitted.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values(by="importance", ascending=False)

print(feature_importance.head(20))

selected_features = ["Income_Median", "Higher_Ed_Ratio"]
selected_feature_importance = feature_importance[
    feature_importance["feature"].str.replace(r"^[^_]+__", "", regex=True).isin(selected_features)
]

print("Income_Median / Higher_Ed_Ratio feature importance：")
print(selected_feature_importance)


# =========================
# 10. 儲存模型
# =========================
joblib.dump(model, "random_forest_housing_model_total_price_time_split.pkl")

print("模型已儲存為 random_forest_housing_model_total_price_time_split.pkl")
