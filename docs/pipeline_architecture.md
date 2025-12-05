# 🔄 BACKTEST PIPELINE ARCHITECTURE

## ⚠️ CSCV vs CPCV vs WALK-FORWARD

**CSCV (Bailey diagnostic)** – Đo PBO/DSR/degradation qua các cấu hình lân cận, KHÔNG bảo toàn thứ tự thời gian (dùng để phát hiện overfitting).  
**CPCV (Purged CV)** – Validation time series có purge/embargo (giữ lại tham chiếu, không còn là đường chính).  
**Purged Walk-Forward Validator** – Kiểm tra thực dụng Y→Y với purge/embargo lấy từ chính `holding_days` giao dịch (health check trước CSCV).

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VALIDATED BACKTEST PIPELINE                   │
├────────────┬──────────────┬────────────────────┬─────────────────────┤
│  CONFIG    │    PRICES    │      PIPELINE      │       RESULT        │
│  (YAML)    │    (CSV)     │  ENGINE + VALIDATE │      (FILES)        │
├────────────┴──────┬───────┴──────────────┬─────┴───────────┬─────────┤
│  WALK-FORWARD     │  PURGED WALK-FWD     │      CSCV       │ OUTPUTS │
│    BACKTEST       │    VALIDATOR         │   DIAGNOSTIC    │ (JSON/  │
│ (formation/trade) │ (health, purge/gap)  │ (PBO/DSR/ranks) │  CSV/TXT│
└───────────────────┴──────────────────────┴─────────────────┴─────────┘
```

## Pipeline Stages

### Stage 0: Load Config
- Đọc `configs/experiments/*.yaml` → `BacktestConfig` (lọc cặp, ngưỡng giao dịch, vốn, output).

### Stage 1: Walk-Forward Backtest (Y-1 formation → Y trading)
- Formation năm Y-1: lọc theo tương quan, Engle–Granger p-value, half-life, SNR ≥ 1.5, ZCR ≥ 5/năm; xếp hạng, chọn top-N; hedge ratio cố định cho năm giao dịch.  
- Trading năm Y: dùng tín hiệu ngày t-1, khớp giá ngày t (NO look-ahead); exit dùng tham số lúc entry (tránh rolling-beta trap); holding động theo half-life; vol sizing; blacklist/stop tightening nếu bật.  
- Kết quả: trades + summary theo năm.

### Stage 1.5: Tính Purge/Embargo từ giao dịch
- `embargo_width = ceil(avg holding_days)`, `purge_width = ceil(max holding_days)` từ log giao dịch; log ra để CV minh bạch.

### Stage 1.6: Purged Walk-Forward Validator (thực dụng)
- Chạy splits train/test có purge/embargo; chấm IS/OOS theo từng split.  
- Mặc định pass: OOS positive ratio ≥ 55%, OOS mean ≥ 0; nếu fail → thêm lỗi/cảnh báo.

### Stage 2: Biến thiên tham số → CSCV (chẩn đoán overfit)
- Sinh lưới tham số (mặc định entry_sigma [1.5,2.0,2.5], exit_sigma [0.0,0.3,0.5] trừ khi override).  
- Chạy backtest cho từng cấu hình, build returns matrix, chạy `CSCVAnalyzer` → PBO, DSR, degradation, rank stability.

### Stage 3: Cổng kiểm tra
- FAIL nếu: PBO > max_pbo, DSR < min_dsr, OOS mean ≤ 0, hoặc walk-forward FAILED.  
- WARN nếu: degradation > 50%, rank yếu, PnL âm, số lệnh thấp.

### Stage 4: Output
- Lưu `trades.csv`, `pipeline_result.json`, `cpcv_report.txt` (CSCV), `validation_summary.txt`, `config_snapshot.yaml` vào `results/<timestamp>_<experiment>/`.

## File Map
- `src/pairs_trading_etf/backtests/pipeline.py` – Orchestrate backtest + walk-forward validator + CSCV.
- `src/pairs_trading_etf/backtests/validation.py` – Purged Walk-Forward Validator.
- `src/pairs_trading_etf/backtests/cpcv_correct.py` – CSCV/CPCV utilities.
- `scripts/run_backtest.py` – CLI chính (chạy toàn bộ pipeline).  
- `scripts/run_cpcv_analysis.py` – CSCV sweep, tùy chọn `--walk-forward`.  
- `scripts/visualize_trade_v2.py` – Visual trade, dùng `config_snapshot.yaml`.

## Usage
- Chuẩn (có validation):  
  `python scripts/run_backtest.py --config configs/experiments/vidyamurthy_practical.yaml --start 2015 --end 2024`
- Nhanh (bỏ diagnostics):  
  `python scripts/run_backtest.py --config configs/experiments/default.yaml --no-cpcv`
- CSCV sweep + walk-forward:  
  `python scripts/run_cpcv_analysis.py --config configs/experiments/vidyamurthy_practical.yaml --sweep --walk-forward`

## Ngưỡng mặc định
- CSCV: PBO < 40%, DSR > 0, OOS mean > 0, degradation < 50%.  
- Walk-forward: OOS positive ratio ≥ 55%, OOS mean ≥ 0 (configurable).

## PipelineConfig (chính)
```
run_cpcv: bool = True
cpcv_n_splits: int = 10
max_pbo: float = 0.40
min_dsr: float = 0.0
require_positive_oos: bool = True
save_results: bool = True
run_walkforward_validator: bool = True
walkforward_train_years: int = 1
walkforward_test_years: int = 1
walkforward_min_positive_ratio: float = 0.55
walkforward_min_oos_return: float = 0.0
walkforward_default_purge: int = 21
walkforward_default_embargo: int = 5
```

*Last Updated: 2025-12-05*
