from predict_nn import predict_and_backtest

res = predict_and_backtest(
    data_path="data/EURUSD_1h_2005-01-01_to_2025-09-23.csv",
    models_dir="models/eurusd_nn"
    
)
print(res["metrics"])  # {'n_trades': ..., 'avg_ret': ..., 'cum_ret': ..., 'sharpe': ..., 'max_dd': ...}
