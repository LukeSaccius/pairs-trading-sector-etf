"""Debug rolling cointegration function."""
import pandas as pd
from pairs_trading_etf.pipelines.rolling_pair_scan import run_rolling_cointegration

prices_df = pd.read_csv('data/raw/etf_prices.csv', index_col='Date', parse_dates=True)
prices = prices_df[['XLU', 'VOO']].dropna()
print(f'Prices shape: {prices.shape}')

try:
    result = run_rolling_cointegration(prices, pairs=[('XLU', 'VOO')], window_days=252, step_days=63)
    print(result)
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
