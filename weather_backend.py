# weather_backend.py
import pandas as pd
from pathlib import Path

# ==== DAILY (dùng module daily.py) ====
from daily import (
    load_daily_artifacts,
    predict_for_date,
    TARGET_COL as DAILY_TARGET_COL,
    HORIZON as DAILY_HORIZON,
)

# ==== HOURLY (dùng module hourly.py) ====
from hourly import (
    load_hourly_artifacts,
    predict_hourly_multi_horizon_for_timestamp,
)

ARTIFACT_DIR = Path("artifacts")

# =========================================================
# 1. LOAD ARTIFACTS DAILY
# =========================================================

# df_daily: sau preprocess, index = datetime
# X_FULL: features đầy đủ (train+val+test) để dùng predict_for_date()
# LGBM_MODELS: dict {h: model} – nằm trong daily, ta không dùng trực tiếp ở đây
# META: meta thông tin (TARGET_COL, HORIZON, feature_cols, ...)
DF_DAILY, X_FULL, _LGBM_MODELS_DAILY, _META_DAILY = load_daily_artifacts(ARTIFACT_DIR)

TARGET_COL = _META_DAILY.get("TARGET_COL", DAILY_TARGET_COL)
HORIZON = _META_DAILY.get("HORIZON", DAILY_HORIZON)
FEATURE_COLS = _META_DAILY.get("feature_cols", X_FULL.columns.tolist())

# bảo đảm sort index
DF_DAILY = DF_DAILY.sort_index()
X_FULL = X_FULL.sort_index()

# =========================================================
# 2. DAILY: HÀM DỰ BÁO CHO UI
# =========================================================

def predict_for_date_lgbm(origin_date, horizon: int = HORIZON):
    """
    Thin wrapper quanh daily.predict_for_date để giữ đúng tên hàm
    cho UI/back-end cũ.
    """
    return predict_for_date(origin_date, horizon=horizon, artifact_dir=ARTIFACT_DIR)


def get_actual_and_forecast_for_ui(origin_date_str, horizon: int = HORIZON):
    """
    Trả về:
    - actual_temp: nhiệt độ thực (daily) tại ngày gốc
    - fc_df: DataFrame dự báo (index = date, cột 'temp_pred')
    - DF_DAILY: full daily DataFrame để UI dùng range chọn ngày
    """
    origin_date = pd.to_datetime(origin_date_str)
    origin_day  = origin_date.normalize()

    # ✅ chỉ cho phép ngày nằm trong VALID_ORIGIN_DATES
    if origin_day not in VALID_ORIGIN_DATES:
        raise ValueError(
            f"Ngày {origin_day.date()} không nằm trong tập ngày hợp lệ để dự báo."
        )

    if origin_date not in DF_DAILY.index:
        raise ValueError(f"Ngày {origin_date.date()} không có trong DF_DAILY.")

    if TARGET_COL not in DF_DAILY.columns:
        raise ValueError(
            f"Không tìm thấy cột target '{TARGET_COL}' trong DF_DAILY."
        )

    actual_temp = float(DF_DAILY.loc[origin_date, TARGET_COL])
    fc_df = predict_for_date_lgbm(origin_date, horizon=horizon)

    return actual_temp, fc_df, DF_DAILY



# =========================================================
# 3. LOAD ARTIFACTS HOURLY
# =========================================================

# df_hourly_clean: hourly sau preprocess, có cột 'datetime' + 'temp'...
# HOURLY_MODELS: dict {h: model}, OHE, META
DF_HOURLY_RAW, HOURLY_MODELS, HOURLY_OHE, HOURLY_META = load_hourly_artifacts(
    ARTIFACT_DIR
)

# Chuẩn hóa DF_HOURLY: index = datetime
DF_HOURLY = DF_HOURLY_RAW.copy() 
if "datetime" in DF_HOURLY.columns:
    DF_HOURLY["datetime"] = pd.to_datetime(DF_HOURLY["datetime"])
    DF_HOURLY = DF_HOURLY.set_index("datetime")
DF_HOURLY = DF_HOURLY.sort_index()


# =========================================================
# 4. HOURLY THỰC TẾ CHO 1 NGÀY
# =========================================================

def get_actual_hourly_for_date(origin_date_str):
    """
    Trả về df_hourly_thật của 1 ngày:
    - index: datetime (giờ trong ngày đó)
    - cột 'temp' (và các cột khác giữ nguyên)
    """
    origin_date = pd.to_datetime(origin_date_str)
    origin_day  = origin_date.normalize()

    if origin_day not in VALID_ORIGIN_DATES:
        raise ValueError(
            f"Ngày {origin_day.date()} không nằm trong tập ngày hợp lệ để hiển thị hourly thực."
        )

    if DF_HOURLY is None or DF_HOURLY.empty:
        raise RuntimeError(
            "DF_HOURLY rỗng hoặc chưa được load. Kiểm tra df_hourly_clean.parquet."
        )

    mask = DF_HOURLY.index.date == origin_date.date()
    df_day = DF_HOURLY.loc[mask]

    if df_day.empty:
        raise ValueError(f"Không có dữ liệu hourly cho ngày {origin_date.date()}")

    if "temp" not in df_day.columns:
        raise ValueError("DF_HOURLY không có cột 'temp'.")

    return df_day.copy()




# =========================================================
# 5. HOURLY DỰ BÁO CHO UI
# =========================================================

def get_forecast_hourly_for_timestamp(origin_ts):
    """
    Dự đoán hourly (multi-horizon) cho 1 timestamp cụ thể,
    dùng hàm tiện ích trong hourly.py:
      predict_hourly_multi_horizon_for_timestamp(origin_ts)
    Trả về DataFrame:
      index = datetime (origin_ts + h giờ)
      cột: 'temp_pred'
    """
    return predict_hourly_multi_horizon_for_timestamp(origin_ts, artifact_dir=ARTIFACT_DIR)


def get_forecast_hourly_for_date(origin_date_str):
    """
    Hàm được UI gọi khi bấm vào 1 ô forecast (theo ngày).

    Cách map:
    - Từ origin_date_str (YYYY-MM-DD), tìm timestamp đầu tiên trong DF_HOURLY
      thuộc ngày đó (ví dụ 00:00).
    - Dùng timestamp này làm gốc để dự đoán multi-horizon giờ tới.

    Kết quả:
    - DataFrame index = datetime, cột 'temp_pred'
    """
    origin_date = pd.to_datetime(origin_date_str)

    if DF_HOURLY is None or DF_HOURLY.empty:
        raise RuntimeError(
            "DF_HOURLY rỗng hoặc chưa được load. Kiểm tra df_hourly_clean.parquet."
        )

    # Lấy timestamp đầu tiên trong ngày đó
    mask = DF_HOURLY.index.date == origin_date.date()
    ts_candidates = DF_HOURLY.index[mask]

    if len(ts_candidates) == 0:
        raise ValueError(
            f"Không tìm thấy timestamp hourly nào cho ngày {origin_date.date()} "
            "để làm gốc dự báo."
        )

    origin_ts = ts_candidates[0]  # ví dụ: 00:00 của ngày đó
    df_pred = get_forecast_hourly_for_timestamp(origin_ts)

    return df_pred

# =========================================================
# 3.5. TẬP NGÀY HỢP LỆ CHO UI
# =========================================================

def _to_date_index(idx):
    """
    Chuyển index datetime -> Index các ngày (00:00), unique, đã sort.
    """
    idx = pd.to_datetime(idx)
    return pd.Index(idx.normalize().unique()).sort_values()

# Các ngày tồn tại trong từng nguồn
_DAILY_DATES   = _to_date_index(DF_DAILY.index)
_FEATURE_DATES = _to_date_index(X_FULL.index)
_HOURLY_DATES  = _to_date_index(DF_HOURLY.index)

# Chỉ những ngày có đủ:
# - daily thực
# - feature cho model daily
# - dữ liệu hourly thực
VALID_ORIGIN_DATES = (
    _DAILY_DATES
    .intersection(_FEATURE_DATES)
).sort_values()


def get_origin_date_range_for_ui():
    """
    Hàm UI sẽ gọi để set min/max cho date_input.
    Trả về (min_date, max_date) dạng date().
    """
    if len(VALID_ORIGIN_DATES) == 0:
        raise RuntimeError("Không có ngày hợp lệ nào trong VALID_ORIGIN_DATES.")
    return VALID_ORIGIN_DATES[0].date(), VALID_ORIGIN_DATES[-1].date()
