# hourly.py — module dùng cho backend UI
# - CHUẨN TÁCH TRAIN/VAL/TEST
# - TRAIN CHỈ TRÊN TRAIN
# - LightGBM multi-horizon
# - Có build_and_save_hourly_models() + load_hourly_artifacts() + predict_hourly_multi_horizon_for_timestamp()

import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ===========================================================
# CẤU HÌNH CHUNG
# ===========================================================
RANDOM_STATE = 42
TARGET_COL   = "temp"
HORIZON      = [1, 6, 12, 23]          # horizon theo giờ
LAGS         = [1, 2, 3, 6, 24]        # lags theo giờ
ROLL_WINDOWS = [3, 6, 12, 24]          # rolling window theo giờ

LAG_COLS = [
    "humidity", "dew", "precip", "precipprob", "solarradiation",
    "sealevelpressure", "windspeed", "winddir", "windgust",
    "cloudcover", "visibility"
]
ROLL_COLS = LAG_COLS.copy()

# Thư mục chứa chính file .py này
BASE_DIR = Path(__file__).resolve().parent

# Folder artifacts nằm cùng cấp với daily.py / hourly.py
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

DEFAULT_HOURLY_SOURCE = (
    "https://raw.githubusercontent.com/DanhBitoo/-Group-2-Machine-Learning-Project/refs/heads/main/Data/Hanoi%20Hourly.csv"
)

# ===========================================================
# 1. LOAD & PREPROCESS HOURLY
# ===========================================================

def load_raw_hourly(path_or_url: str | None = None) -> pd.DataFrame:
    """
    Load dữ liệu hourly gốc.
    Mặc định: dùng URL GitHub (bạn có thể đổi sang file local nếu muốn).
    """
    if path_or_url is None:
        path_or_url = DEFAULT_HOURLY_SOURCE
    df = pd.read_csv(path_or_url)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["datetime"].dt.month

    def get_season(m: int) -> str:
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)
    return df


def impute_wind_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    - FFill windspeed theo từng 'name' (trạm)
    - Điền winddir theo mode trong group (name, season, hour)
    """
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour

    # Forward-fill windspeed theo từng name (trạm)
    def ffill_windspeed(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("datetime")
        group["windspeed"] = group["windspeed"].ffill()
        return group

    if "name" in df.columns:
        df = df.groupby("name", group_keys=False).apply(ffill_windspeed)
    else:
        df["windspeed"] = df["windspeed"].ffill()

    # Điền winddir bằng mode trong group (name, season, hour)
    group_cols = []
    if "name" in df.columns:
        group_cols.append("name")
    if "season" in df.columns:
        group_cols.append("season")
    group_cols.append("hour")

    if "winddir" in df.columns and group_cols:
        df["winddir"] = df.groupby(group_cols)["winddir"].transform(
            lambda x: x.fillna(
                x.mode()[0] if not x.mode().empty else x.median()
            )
        )

    return df


def impute_precip_visibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    - precip: fill 0
    - visibility: ffill rồi bfill
    """
    df = df.copy()
    if "precip" in df.columns:
        df["precip"] = df["precip"].fillna(0.0)
    if "visibility" in df.columns:
        df["visibility"] = df["visibility"].ffill()
        df["visibility"] = df["visibility"].bfill()
    return df


def impute_solar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute cho solarradiation / solarenergy / uvindex:
    - Ban đêm (18-5h): nếu NaN thì gán 0
    - Ban ngày: dùng median theo bin cloudcover + hour
    - Còn lại: fill median
    """
    df = df.copy()

    if not {"solarradiation", "solarenergy", "uvindex", "cloudcover"}.issubset(df.columns):
        return df

    df["hour"] = df["datetime"].dt.hour
    df["is_night"] = df["hour"].isin(range(18, 24)) | df["hour"].isin(range(0, 6))

    # Ban đêm: NaN -> 0
    night_mask = df["is_night"] & df["solarradiation"].isna()
    df.loc[night_mask, ["solarradiation", "solarenergy", "uvindex"]] = 0.0

    # Bin cloudcover
    df["cloudcover_bin"] = pd.cut(
        df["cloudcover"],
        bins=[0, 30, 60, 100],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    day_mask = ~df["is_night"]
    vars_solar = ["solarradiation", "solarenergy", "uvindex"]

    # median theo (cloudcover_bin, hour) cho ban ngày
    group = df[day_mask].groupby(["cloudcover_bin", "hour"], dropna=False)
    for col in vars_solar:
        medians = group[col].transform("median")
        idx = day_mask & df[col].isna()
        df.loc[idx, col] = medians[idx]

    # fallback: median toàn cột
    for col in vars_solar:
        df[col] = df[col].fillna(df[col].median())

    # cleanup
    df = df.drop(columns=["cloudcover_bin", "is_night"], errors="ignore")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    df = df.drop(columns=constant_cols, errors="ignore")
    return df


def preprocess_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp toàn bộ bước xử lý missing & cột mới ở trên.
    """
    df = df.copy()
    df = add_season_column(df)
    df = impute_wind_fields(df)
    df = impute_precip_visibility(df)
    df = impute_solar_variables(df)

    # flag mưa
    if "precip" in df.columns:
        df["precip_flag"] = (df["precip"] > 0).astype(int)

    # bỏ vài cột không cần (nếu có)
    drop_cols = ["severerisk", "preciptype", "snow", "snowdepth", "stations"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = drop_constant_columns(df)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ===========================================================
# 2. ONE-HOT ENCODING CHO CATEGORICAL ('icon', 'season')
# ===========================================================

def fit_ohe(train_df: pd.DataFrame) -> OneHotEncoder | None:
    cols = [c for c in ["icon", "season"] if c in train_df.columns]
    if not cols:
        return None
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(train_df[cols])
    return ohe


def transform_ohe(df: pd.DataFrame, ohe: OneHotEncoder | None) -> pd.DataFrame:
    if ohe is None:
        return df
    df = df.copy()
    cols = [c for c in ["icon", "season"] if c in df.columns]
    encoded = ohe.transform(df[cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cols),
        index=df.index
    )
    df = pd.concat([df.drop(columns=cols, errors="ignore"), encoded_df], axis=1)
    return df


# ===========================================================
# 3. FEATURE ENGINEERING (LAG / ROLL / CYCLICAL)
# ===========================================================

def encode_cyclical(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def _create_hourly_features_base(
    df: pd.DataFrame,
    target_col: str,
    horizon: list[int],
    lag_features: list[str],
    lags: list[int],
    roll_features: list[str],
    windows: list[int],
    include_targets: bool,
) -> pd.DataFrame:
    df_new = df.copy()
    df_new = df_new.sort_values("datetime").set_index("datetime")

    # 1. time features
    df_new["hour"] = df_new.index.hour
    df_new["day_of_week"] = df_new.index.dayofweek
    df_new["day_of_year"] = df_new.index.dayofyear
    df_new["month"] = df_new.index.month

    # 2. cyclical encoding
    df_new = encode_cyclical(df_new, "hour", 24)
    df_new = encode_cyclical(df_new, "day_of_week", 7)
    df_new = encode_cyclical(df_new, "day_of_year", 366)
    df_new = encode_cyclical(df_new, "month", 12)
    if "winddir" in df_new.columns:
        df_new = encode_cyclical(df_new, "winddir", 360)
    df_new = df_new.drop(
        columns=["hour", "day_of_week", "day_of_year", "month", "winddir"],
        errors="ignore"
    )

    # 3. lags
    for feature in lag_features:
        if feature in df_new.columns:
            for lag in lags:
                df_new[f"{feature}_lag{lag}"] = df_new[feature].shift(lag)

    # 4. rolling statistics
    for feature in roll_features:
        if feature in df_new.columns:
            for w in windows:
                rolling_window = df_new[feature].rolling(window=w)
                df_new[f"{feature}_roll{w}_mean"] = rolling_window.mean()
                df_new[f"{feature}_roll{w}_std"] = rolling_window.std()

    # 5. derived features
    if {"temp", "dew"}.issubset(df_new.columns):
        df_new["dewpoint_depression"] = df_new["temp"] - df_new["dew"]
    if "windspeed" in df_new.columns:
        df_new["wind_speed_squared"] = df_new["windspeed"] ** 2
    if "humidity" in df_new.columns:
        df_new["humidity_ratio"] = df_new["humidity"] / 100.0
    if {"temp", "windspeed"}.issubset(df_new.columns):
        df_new["wind_chill"] = df_new["temp"] - 0.1 * df_new["windspeed"]
    if {"windspeed", "windgust"}.issubset(df_new.columns):
        df_new["wind_ratio"] = df_new["windgust"] / (df_new["windspeed"] + 1e-6)
    if {"precipprob", "precip_flag"}.issubset(df_new.columns):
        df_new["severe_proxy"] = df_new["precipprob"] * df_new["precip_flag"]
    if {"temp", "humidity", "wind_speed_squared"}.issubset(df_new.columns):
        df_new["heat_index_approx"] = (
            df_new["temp"]
            + 0.33 * df_new["humidity"]
            - 0.70 * np.sqrt(df_new["wind_speed_squared"])
            - 4.00
        )

    # 6. targets (chỉ khi train)
    if include_targets:
        for h in horizon:
            df_new[f"target_{target_col}_t+{h}"] = df_new[target_col].shift(-h)

    # 7. drop temp gốc khỏi features
    df_new = df_new.drop(columns=[target_col], errors="ignore")

    # 8. dropna:
    #    - TRAIN: drop (cần đủ lag/rolling/target)
    #    - INFERENCE: giữ NaN, LightGBM tự xử lý được
    if include_targets:
        df_new = df_new.dropna()

    return df_new


def create_hourly_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: list[int] = HORIZON,
    lag_features: list[str] = LAG_COLS,
    lags: list[int] = LAGS,
    roll_features: list[str] = ROLL_COLS,
    windows: list[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Dùng cho TRAIN: tạo cả target_{target_col}_t+{h} và dropna.
    """
    return _create_hourly_features_base(
        df,
        target_col=target_col,
        horizon=horizon,
        lag_features=lag_features,
        lags=lags,
        roll_features=roll_features,
        windows=windows,
        include_targets=True,
    )


def create_hourly_features_for_predictions(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: list[int] = HORIZON,
    lag_features: list[str] = LAG_COLS,
    lags: list[int] = LAGS,
    roll_features: list[str] = ROLL_COLS,
    windows: list[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Dùng cho INFERENCE:
    - KHÔNG tạo cột target
    - KHÔNG dropna (row đầu có thể thiếu lag/rolling, LightGBM vẫn ăn được)
    """
    return _create_hourly_features_base(
        df,
        target_col=target_col,
        horizon=horizon,
        lag_features=lag_features,
        lags=lags,
        roll_features=roll_features,
        windows=windows,
        include_targets=False,
    )


# ===========================================================
# 4. SPLIT TRAIN / VAL / TEST
# ===========================================================

def split_train_val_test(df_base: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """
    Split theo thời gian: train 70% - val 15% - test 15%.
    """
    n = len(df_base)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df_base.iloc[:train_end].copy()
    val_df = df_base.iloc[train_end:val_end].copy()
    test_df = df_base.iloc[val_end:].copy()
    return train_df, val_df, test_df


# ===========================================================
# 5. TRAIN LIGHTGBM MULTI-HORIZON
# ===========================================================

def train_lgbm_hourly_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    horizon: list[int] = HORIZON,
    target_col: str = TARGET_COL,
):
    """
    Train 1 LGBMRegressor cho mỗi horizon.
    Trả về:
        - models: dict {h: model}
        - metrics: dict {h: {'val_mae', 'val_rmse', 'val_r2', 'test_mae', ...}}
    """
    models = {}
    metrics = {}
    for h in horizon:
        target_name = f"target_{target_col}_t+{h}"

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="regression",
        )

        model.fit(X_train, y_train[target_name])

        # evaluate (không print, chỉ trả về)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        metrics[h] = {
            "val_mae": mean_absolute_error(y_val[target_name], y_pred_val),
            "val_rmse": np.sqrt(
                mean_squared_error(y_val[target_name], y_pred_val)
            ),
            "val_r2": r2_score(y_val[target_name], y_pred_val),
            "test_mae": mean_absolute_error(y_test[target_name], y_pred_test),
            "test_rmse": np.sqrt(
                mean_squared_error(y_test[target_name], y_pred_test)
            ),
            "test_r2": r2_score(y_test[target_name], y_pred_test),
        }

        models[h] = model

    return models, metrics


# ===========================================================
# 6. PIPELINE HOÀN CHỈNH + LƯU ARTIFACT
# ===========================================================

def build_and_save_hourly_models(
    source: str | None = None,
    artifact_dir: Path = ARTIFACT_DIR,
):
    """
    Chạy full pipeline:
    - Load & preprocess df_hourly
    - Drop một số cột text (stations, conditions, feelslike)
    - Split train/val/test
    - OHE
    - FE (lags/roll/cyclical + targets)
    - Train LGBM multi-horizon
    - Lưu:
        - df_hourly_clean.parquet
        - lgbm_hourly_models.pkl
        - hourly_ohe.pkl
        - hourly_meta.pkl
    """
    df_raw = load_raw_hourly(source)

    # bỏ vài cột text không dùng (nếu còn)
    for col in ["stations", "conditions", "feelslike"]:
        if col in df_raw.columns:
            df_raw.drop(columns=col, inplace=True)

    df_clean = preprocess_hourly(df_raw)

    df_base = df_clean.copy()
    train_df, val_df, test_df = split_train_val_test(df_base)

    # One-hot encoding
    ohe = fit_ohe(train_df)
    train_df = transform_ohe(train_df, ohe)
    val_df = transform_ohe(val_df, ohe)
    test_df = transform_ohe(test_df, ohe)

    # Feature engineering (có target)
    df_train_fe = create_hourly_features(train_df)
    df_val_fe = create_hourly_features(val_df)
    df_test_fe = create_hourly_features(test_df)

    target_cols_full = [f"target_{TARGET_COL}_t+{h}" for h in HORIZON]

    X_train = df_train_fe.drop(columns=target_cols_full)
    y_train = df_train_fe[target_cols_full]
    X_val = df_val_fe.drop(columns=target_cols_full)
    y_val = df_val_fe[target_cols_full]
    X_test = df_test_fe.drop(columns=target_cols_full)
    y_test = df_test_fe[target_cols_full]

    models, metrics = train_lgbm_hourly_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Lưu artifact
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # df_hourly_clean
    df_clean.to_parquet(artifact_dir / "df_hourly_clean.parquet", index=False)

    joblib.dump(models, artifact_dir / "lgbm_hourly_models.pkl")
    joblib.dump(ohe, artifact_dir / "hourly_ohe.pkl")

    meta = {
        "TARGET_COL": TARGET_COL,
        "HORIZON": HORIZON,
        "LAG_COLS": LAG_COLS,
        "ROLL_COLS": ROLL_COLS,
        "LAGS": LAGS,
        "ROLL_WINDOWS": ROLL_WINDOWS,
        "feature_cols": X_train.columns.tolist(),
        "metrics": metrics,
    }
    joblib.dump(meta, artifact_dir / "hourly_meta.pkl")


# ===========================================================
# 7. HÀM PREDICT HỖ TRỢ UI (DỰ BÁO NHIỀU GIỜ TƯƠNG LAI)
# ===========================================================

def load_hourly_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    df_clean = pd.read_parquet(artifact_dir / "df_hourly_clean.parquet")
    models = joblib.load(artifact_dir / "lgbm_hourly_models.pkl")
    ohe = joblib.load(artifact_dir / "hourly_ohe.pkl")
    meta = joblib.load(artifact_dir / "hourly_meta.pkl")
    return df_clean, models, ohe, meta


def predict_hourly_multi_horizon_for_timestamp(
    origin_ts,
    artifact_dir: Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    """
    Hàm tiện ích cho UI:
    - origin_ts: một timestamp (datetime-like) làm gốc (phải nằm trong df_hourly_clean)
    - Trả về DataFrame:
        index: thời điểm dự báo (origin_ts + h giờ)
        cột: 'temp_pred'

    Dùng feature_space giống lúc train:
      - transform_ohe
      - create_hourly_features_for_predictions (KHÔNG dropna)
    """
    df_clean, models, ohe, meta = load_hourly_artifacts(artifact_dir)
    horizon = meta["HORIZON"]
    feature_cols = meta["feature_cols"]

    df_clean = df_clean.copy()
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"])
    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)

    origin_ts = pd.to_datetime(origin_ts)

    # Kiểm tra origin_ts có nằm trong df_clean hay không
    if origin_ts < df_clean["datetime"].min() or origin_ts > df_clean["datetime"].max():
        raise ValueError(
            f"origin_ts {origin_ts} nằm ngoài khoảng dữ liệu hourly."
        )

    # apply OHE
    df_clean_ohe = transform_ohe(df_clean, ohe)

    # tạo full features cho prediction (KHÔNG dropna)
    df_feat = create_hourly_features_for_predictions(df_clean_ohe)

    if origin_ts not in df_feat.index:
        raise ValueError(
            f"origin_ts {origin_ts} không nằm trong index features sau FE. "
            f"Bạn có thể cần chọn timestamp muộn hơn một chút để đủ dữ liệu lag/rolling."
        )

    X_input = df_feat.loc[[origin_ts], feature_cols]

    preds_times = []
    preds_vals = []
    for h in horizon:
        model = models[h]
        y_hat = float(model.predict(X_input)[0])
        forecast_time = origin_ts + pd.Timedelta(hours=h)
        preds_times.append(forecast_time)
        preds_vals.append(y_hat)

    pred_df = pd.DataFrame(
        {"datetime": preds_times, "temp_pred": preds_vals}
    ).set_index("datetime")

    return pred_df


# ===========================================================
# MAIN (nếu chạy file trực tiếp để train & lưu artifact)
# ===========================================================
if __name__ == "__main__":
    build_and_save_hourly_models()
