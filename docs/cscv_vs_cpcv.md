# CSCV vs CPCV: Hướng dẫn cho Pairs Trading

## ⚠️ QUAN TRỌNG: CHÚNG HOÀN TOÀN KHÁC NHAU!

| Tiêu chí | CSCV | CPCV |
|----------|------|------|
| **Tên đầy đủ** | Combinatorial Symmetric CV | Combinatorial Purged CV |
| **Dùng cho** | Đo mức độ overfit | Validation time series |
| **Thử tự thời gian** | ❌ KHÔNG | ✅ CÓ |
| **Purging** | ❌ KHÔNG | ✅ CÓ |
| **Embargo** | ❌ KHÔNG | ✅ CÓ |
| **Data leak** | 🔴 CÓ (cố ý) | 🟢 KHÔNG |
| **Safe cho trading** | ❌ KHÔNG | ✅ CÓ |

---

## 1. CSCV (Combinatorial Symmetric Cross-Validation)

### Mục đích
- **Đo lường** mức độ overfitting
- **KHÔNG dùng để validate** trading strategy

### Cách hoạt động (SAI cho time series!)
```
Data: [Jan, Feb, Mar, Apr, May, Jun]
       [0]  [1]  [2]  [3]  [4]  [5]

CSCV thử TẤT CẢ C(6,3) = 20 combinations:
  Combo 1: Train [0,1,2] | Test [3,4,5]  ✓ OK
  Combo 2: Train [0,1,3] | Test [2,4,5]  ✗ Apr dùng để predict Mar!
  Combo 3: Train [0,2,4] | Test [1,3,5]  ✗ Mixing time completely!
  ...

VẤN ĐỀ: Future data leak vào training!
```

### Khi nào dùng CSCV?
- ✅ Đo mức độ overfitting của strategy (so sánh PBO CSCV vs CPCV)
- ✅ Data i.i.d. (images, text)
- ❌ KHÔNG dùng cho stock price prediction
- ❌ KHÔNG dùng cho pairs trading validation

---

## 2. CPCV (Combinatorial Purged Cross-Validation)

### Mục đích
- **Validate** trading strategy với temporal ordering
- Đảm bảo train TRƯỚC test (không future leak)

### Cách hoạt động (ĐÚNG cho time series)
```
Data: [Jan, Feb, Mar, Apr, May, Jun]
       [0]  [1]  [2]  [3]  [4]  [5]

CPCV chỉ thử TEMPORALLY VALID splits:
  Split 1: Train [0]     | Test [1,2,3]    ✓ Train trước Test
  Split 2: Train [0,1]   | Test [2,3,4]    ✓ Train trước Test
  Split 3: Train [0,1,2] | Test [3,4,5]    ✓ Train trước Test

KHÔNG BAO GIỜ:
  ❌ Train [0,3] | Test [1,2]  (Apr trong train, nhưng predict Feb/Mar)
  ❌ Train [2,4] | Test [1,3]  (non-contiguous, mixing)
```

### Purge & Embargo
```
|---- Train ----|--Purge--|---- Test ----|--Embargo--|

Purge: Bỏ data cuối train (tránh overlap trades)
       Rule: purge_window = ceil(max_holding_period)

Embargo: Gap trước train tiếp theo (market adjustment)
         Rule: embargo_window = ceil(avg_holding_period)
```

---

## 3. Walk-Forward CPCV (Tốt nhất cho Pairs Trading)

### Cách hoạt động
```
Year 2010: Train (formation period)
           ↓ Purge (1 tháng)
Year 2011: Test (trading period)
           ↓ Embargo (2 tuần)
Year 2012: Train mới
           ↓ Purge
Year 2013: Test
...
```

### Implementation trong project
```python
from pairs_trading_etf.backtests.cpcv_correct import WalkForwardCPCV

wf = WalkForwardCPCV(
    train_years=1,      # 1 năm formation
    test_years=1,       # 1 năm trading
    purge_days=21,      # ~1 tháng purge
    embargo_days=10,    # ~2 tuần embargo (= avg holding)
)

result = wf.analyze(returns_matrix, dates, strategy_names)
```

---

## 4. So sánh kết quả thực tế

Với synthetic data (10 năm, 9 strategies):

```
┌─────────────────────────────────────────────────────────┐
│                    PBO COMPARISON                        │
├─────────────────────────────────────────────────────────┤
│  CSCV PBO:         30.6%  ← Có data leak, quá lạc quan  │
│  CPCV PBO:         80.0%  ← Proper ordering, thực tế    │
│  Walk-Forward PBO: 33.3%  ← Realistic trading scenario  │
└─────────────────────────────────────────────────────────┘

Interpretation:
- CSCV says: "Only 30% chance of overfit" (FALSE!)
- CPCV says: "80% chance best IS fails OOS" (TRUE)
- Walk-Forward: Matches real trading with re-calibration
```

---

## 5. Files trong project

```
src/pairs_trading_etf/backtests/
├── cpcv.py           # OLD implementation (có lỗi logic)
├── cpcv_correct.py   # NEW correct implementation ✓
│   ├── CPCVAnalyzer      # Proper temporal CPCV
│   ├── WalkForwardCPCV   # Best for pairs trading
│   └── CSCVAnalyzer      # For overfitting detection only
└── pipeline.py       # Integrated pipeline
```

---

## 6. Checklist cho Pairs Trading

1. ☐ Dùng `WalkForwardCPCV` cho validation (KHÔNG dùng CSCV)
2. ☐ Set `purge_days >= max_holding_period` 
3. ☐ Set `embargo_days >= avg_holding_period`
4. ☐ Check PBO < 40% (MODERATE risk)
5. ☐ Check degradation < 50%
6. ☐ Compare CSCV vs CPCV để thấy mức overfitting

---

## 7. Tài liệu tham khảo

- Bailey et al. (2016) - "The Probability of Backtest Overfitting"
- López de Prado (2018) - "Advances in Financial Machine Learning" Ch.7
- Harvey et al. (2016) - "... and the Cross-section of Expected Returns"

---

*Last Updated: December 4, 2025*
