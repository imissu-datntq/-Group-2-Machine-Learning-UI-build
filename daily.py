# daily.py — phiên bản module hóa, dùng cho backend UI
# - CHUẨN TÁCH TRAIN/VAL/TEST
# - TRAIN CHỈ TRÊN TRAIN
# - Không tuning, không print log
# - Có build_and_save_daily_models() + load_daily_artifacts() + predict_for_date()

import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
import warnings
import joblib

# =======================================================
# CONFIG
# =======================================================

TARGET_COL = "temp"
HORIZON = 5
RANDOM_STATE = 42

LAGS = [1, 3, 5, 7]
ROLL_WINDOWS = [7, 14, 28, 56, 84]

LAG_FEATURES = [
    "humidity",
    "dew",
    "precip",
    "precipprob",
    "precipcover",
    "solarradiation",
    "sealevelpressure",
    "windspeed",
    "winddir",
    "windgust",
    "cloudcover",
    "visibility",
]

ROLL_FEATURES = LAG_FEATURES.copy()

# Thư mục chứa chính file .py này
BASE_DIR = Path(__file__).resolve().parent

# Folder artifacts nằm cùng cấp với daily.py / hourly.py
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_DAILY_SOURCE = (
    "https://raw.githubusercontent.com/DanhBitoo/-Group-2-Machine-Learning-Project/refs/heads/main/Data/Hanoi%20Daily.csv"
)

# =======================================================
# 1. LOAD & PREPROCESS DAILY
# =======================================================

def load_daily_raw(path_or_url: str | None = None) -> pd.DataFrame:
    """
    Load dữ liệu daily gốc.
    Mặc định: dùng URL GitHub (bạn có thể đổi sang file local nếu muốn).
    """
    if path_or_url is None:
        path_or_url = DEFAULT_DAILY_SOURCE
    df = pd.read_csv(path_or_url)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def preprocess_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu:
    - Parse datetime / sunrise / sunset
    - Sort & set index datetime
    - Xử lý preciptype dạng numeric (rain/none)
    - Drop severerisk, constant cols, các cột text / không phù hợp
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["sunrise"] = pd.to_datetime(df.get("sunrise"), errors="coerce")
    df["sunset"] = pd.to_datetime(df.get("sunset"), errors="coerce")

    df = df.sort_values("datetime").set_index("datetime")

    # preciptype -> numeric (rain/none)
    if "preciptype" in df.columns:
        df["preciptype"] = df["preciptype"].fillna("none")
        df["preciptype"] = df["preciptype"].replace({"rain": 1, "none": 0})
        df["preciptype"] = pd.to_numeric(df["preciptype"], downcast="integer")

    # Drop severerisk nếu có
    df = df.drop(columns=[c for c in ["severerisk"] if c in df.columns], errors="ignore")

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols, errors="ignore")

    # Drop các cột text / không phù hợp (kể cả 'name')
    problem_cols = [
        "name",
        "sunrise",
        "sunset",
        "conditions",
        "description",
        "icon",
        "stations",
    ]
    existing_problem_cols = [col for col in problem_cols if col in df.columns]
    df = df.drop(columns=existing_problem_cols, errors="ignore")

    return df


# =======================================================
# 2. TÁCH TRAIN / VAL / TEST THEO THỜI GIAN
# =======================================================

def split_train_val_test_daily(
    df_daily: pd.DataFrame, train_ratio=0.7, val_ratio=0.15
):
    """
    Split theo index thời gian: train 70% - val 15% - test 15%
    """
    n = len(df_daily)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df_daily.iloc[:train_end].copy()
    val_df = df_daily.iloc[train_end:val_end].copy()
    test_df = df_daily.iloc[val_end:].copy()

    return train_df, val_df, test_df


# =======================================================
# 3. FEATURE ENGINEERING
# =======================================================

def encode_cyclical(df: pd.DataFrame, col: str, max_val: int):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def fe_transform(
    df: pd.DataFrame,
    lag_feats: list[str] | None = None,
    roll_feats: list[str] | None = None,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    FE:
    - time features
    - cyclical (month, day_of_year, day_of_week)
    - lag features
    - rolling features
    - derived features
    """
    if lag_feats is None:
        lag_feats = [c for c in LAG_FEATURES if c in df.columns]
    if roll_feats is None:
        roll_feats = [c for c in ROLL_FEATURES if c in df.columns]
    if lags is None:
        lags = LAGS
    if roll_windows is None:
        roll_windows = ROLL_WINDOWS

    df = df.copy()

    # time features
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["day_of_week"] = df.index.dayofweek
    df["quarter"] = df.index.quarter

    df = encode_cyclical(df, "month", 12)
    df = encode_cyclical(df, "day_of_year", 366)
    df = encode_cyclical(df, "day_of_week", 7)

    # lag features
    lag_features = []
    for col in lag_feats:
        for L in lags:
            lag_features.append(df[col].shift(L).rename(f"{col}_lag{L}"))

    # rolling features
    roll_features = []
    for col in roll_feats:
        for w in roll_windows:
            r = df[col].rolling(w)
            roll_features.append(r.mean().rename(f"{col}_roll{w}_mean"))
            roll_features.append(r.std().rename(f"{col}_roll{w}_std"))

    if lag_features:
        df = pd.concat([df] + lag_features, axis=1)
    if roll_features:
        df = pd.concat([df] + roll_features, axis=1)

    # derived features
    derived_features = []
    if {"tempmax", "tempmin"}.issubset(df.columns):
        derived_features.append((df["tempmax"] - df["tempmin"]).rename("temp_range"))

    if {"temp", "dew"}.issubset(df.columns):
        derived_features.append((df["temp"] - df["dew"]).rename("dewpoint_depression"))

    if derived_features:
        df = pd.concat([df] + derived_features, axis=1)

    return df


def fe_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE cho mục đích dự báo:
    - Dùng cùng logic fe_transform (time, lag, rolling, derived)
    - KHÔNG tạo target, KHÔNG dropna
    - Chỉ giữ numeric/bool cho LightGBM
    """
    df_feat = fe_transform(df)

    # chỉ lấy numeric/bool
    X_inf = df_feat.select_dtypes(include=["number", "bool"])

    # giữ index datetime để tra cứu theo ngày
    X_inf = X_inf.sort_index()
    return X_inf


def fe_with_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: int = HORIZON,
    lag_feats: list[str] | None = None,
    roll_feats: list[str] | None = None,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Trả về:
    - X: DataFrame features numeric/bool
    - y: dict {h: Series target_{target_col}_t+{h}}
    """
    df = fe_transform(df, lag_feats, roll_feats, lags, roll_windows)
    df_feat = df.copy()

    # tạo target
    for h in range(1, horizon + 1):
        df_feat[f"target_{target_col}_t+{h}"] = df_feat[target_col].shift(-h)

    df_feat = df_feat.dropna()

    target_cols = [f"target_{target_col}_t+{h}" for h in range(1, horizon + 1)]
    X_raw = df_feat.drop(columns=target_cols)

    # chỉ giữ numeric/bool để tránh lỗi dtype
    X = X_raw.select_dtypes(include=["number", "bool"])

    y = {}
    for h in range(1, horizon + 1):
        col = f"target_{target_col}_t+{h}"
        # align index với X
        y[h] = df_feat[col].loc[X.index]

    return X, y


# =======================================================
# 4. TRAIN LIGHTGBM MULTI-HORIZON
# =======================================================

def train_lgbm_daily_models(
    X_train: pd.DataFrame,
    y_train: dict,
    horizon: int = HORIZON,
    random_state: int = RANDOM_STATE,
):
    """
    Train 1 LGBMRegressor cho mỗi horizon.
    Trả về dict: {h: model}
    """
    models_daily = {}
    for h in range(1, horizon + 1):
        model = LGBMRegressor(
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        model.fit(X_train, y_train[h])
        models_daily[h] = model
    return models_daily


# =======================================================
# 5. PIPELINE HOÀN CHỈNH + LƯU ARTIFACT
# =======================================================

def build_and_save_daily_models(artifact_dir: Path = ARTIFACT_DIR):
    """
    Chạy full pipeline:
    - Load & preprocess daily
    - Split train/val/test theo thời gian
    - FE train/val/test
    - Train LGBM multi-horizon (CHỈ TRÊN TRAIN)
    - Tạo X_full (train+val+test) để backend dùng nếu cần
    - Lưu artifacts:
        - df_daily.parquet
        - X_features.parquet
        - lgbm_models.pkl
        - meta.pkl
    """
    # 1) Load & preprocess
    df_raw = load_daily_raw()
    df_daily = preprocess_daily(df_raw)

    # 2) Split
    train_df, val_df, test_df = split_train_val_test_daily(df_daily)

    # 3) FE (train/val/test)
    lag_feats = [c for c in LAG_FEATURES if c in df_daily.columns]
    roll_feats = [c for c in ROLL_FEATURES if c in df_daily.columns]

    X_train, y_train = fe_with_target(
        train_df,
        target_col=TARGET_COL,
        horizon=HORIZON,
        lag_feats=lag_feats,
        roll_feats=roll_feats,
        lags=LAGS,
        roll_windows=ROLL_WINDOWS,
    )
    X_val, y_val = fe_with_target(
        val_df,
        target_col=TARGET_COL,
        horizon=HORIZON,
        lag_feats=lag_feats,
        roll_feats=roll_feats,
        lags=LAGS,
        roll_windows=ROLL_WINDOWS,
    )
    X_test, y_test = fe_with_target(
        test_df,
        target_col=TARGET_COL,
        horizon=HORIZON,
        lag_feats=lag_feats,
        roll_feats=roll_feats,
        lags=LAGS,
        roll_windows=ROLL_WINDOWS,
    )

    # 4) Ghép X_full (train+val+test) — dùng cho backend nếu cần
    X_full = pd.concat([X_train, X_val, X_test]).sort_index()

    # 5) Train models (chỉ trên train)
    models_daily = train_lgbm_daily_models(X_train, y_train, horizon=HORIZON)

    # 6) Lưu artifacts
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # df_daily sau preprocess
    df_daily.to_parquet(artifact_dir / "df_daily.parquet")

    # X_full để backend dùng (range, debug, v.v.)
    X_full.to_parquet(artifact_dir / "X_features.parquet")

    # models
    joblib.dump(models_daily, artifact_dir / "lgbm_models.pkl")

    # meta
    meta = {
        "TARGET_COL": TARGET_COL,
        "HORIZON": HORIZON,
        "LAGS": LAGS,
        "ROLL_WINDOWS": ROLL_WINDOWS,
        "LAG_FEATURES": LAG_FEATURES,
        "ROLL_FEATURES": ROLL_FEATURES,
        "feature_cols": X_full.columns.tolist(),
    }
    joblib.dump(meta, artifact_dir / "meta.pkl")


# =======================================================
# 6. HÀM LOAD ARTIFACTS + PREDICT HỖ TRỢ UI
# =======================================================

def load_daily_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    """
    Load toàn bộ artifacts đã lưu:
    - df_daily: daily sau preprocess, index = datetime
    - X_full: features train+val+test (chủ yếu cho backend/UI dùng range)
    - models_daily: dict {h: model}
    - meta: dict (TARGET_COL, HORIZON, feature_cols, ...)
    """
    df_daily = pd.read_parquet(artifact_dir / "df_daily.parquet")
    df_daily = df_daily.sort_index()

    X_full = pd.read_parquet(artifact_dir / "X_features.parquet")
    X_full = X_full.sort_index()

    models_daily = joblib.load(artifact_dir / "lgbm_models.pkl")
    meta = joblib.load(artifact_dir / "meta.pkl")

    return df_daily, X_full, models_daily, meta


def predict_for_date(
    origin_date,
    horizon: int = HORIZON,
    artifact_dir: Path = ARTIFACT_DIR,
):
    """
    Dự báo từ một ngày BẤT KỲ trong df_daily (kể cả ngày bị drop khi train).
    Cách làm:
      - Load df_daily + models từ artifacts
      - FE lại toàn bộ df_daily bằng fe_for_inference (KHÔNG dropna, không target)
      - Lấy 1 hàng features tương ứng với origin_date
      - Dùng models_daily[h].predict() cho h=1..HORIZON
    """
    df_daily, X_full, models_daily, meta = load_daily_artifacts(artifact_dir)
    df_daily = df_daily.sort_index()

    origin_date = pd.to_datetime(origin_date)

    if origin_date not in df_daily.index:
        raise ValueError(f"Ngày {origin_date.date()} không có trong df_daily.")

    # FE cho inference toàn bộ chuỗi
    X_infer_full = fe_for_inference(df_daily)

    if origin_date not in X_infer_full.index:
        raise ValueError(
            f"Không tìm thấy features inference cho ngày {origin_date.date()}."
        )

    X_input = X_infer_full.loc[[origin_date]]

    preds = []
    dates = []
    horizon_use = int(meta.get("HORIZON", horizon))

    for h in range(1, horizon_use + 1):
        model = models_daily[h]
        y_hat = float(model.predict(X_input)[0])
        forecast_date = origin_date + pd.Timedelta(days=h)
        preds.append(y_hat)
        dates.append(forecast_date)

    df_pred = pd.DataFrame({"date": dates, "temp_pred": preds}).set_index("date")
    return df_pred


def get_actual_and_forecast_for_ui(
    origin_date,
    horizon: int = HORIZON,
    artifact_dir: Path = ARTIFACT_DIR,
):
    """
    Hàm tiện cho UI:
    - actual_temp: nhiệt độ thực tại origin_date (từ df_daily)
    - fc_df: DataFrame dự báo (1..horizon ngày)
    - df_daily: full daily để UI dùng làm range date
    """
    df_daily, X_full, models_daily, meta = load_daily_artifacts(artifact_dir)
    df_daily = df_daily.sort_index()

    origin_date = pd.to_datetime(origin_date)

    if origin_date not in df_daily.index:
        raise ValueError(f"{origin_date.date()} không có trong df_daily")

    actual_temp = float(df_daily.loc[origin_date, TARGET_COL])
    fc_df = predict_for_date(
        origin_date,
        horizon=int(meta.get("HORIZON", horizon)),
        artifact_dir=artifact_dir,
    )

    return actual_temp, fc_df, df_daily


# =======================================================
# MAIN — TRAIN & LƯU ARTIFACT KHI CHẠY TRỰC TIẾP
# =======================================================

if __name__ == "__main__":
    build_and_save_daily_models()
