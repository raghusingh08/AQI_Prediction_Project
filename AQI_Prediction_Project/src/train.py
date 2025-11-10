import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

from utils import parse_and_sort, add_time_features, make_supervised, train_valid_split_time


def main(args):

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.data)
    df = parse_and_sort(df, args.date_col)

    # Feature engineering
    df = add_time_features(df, args.date_col)
    df = make_supervised(df, target=args.target, max_lag=args.max_lag, roll_windows=(7,14))

    # Move target back (forecasting)
    df[args.target] = df[args.target].shift(-args.horizon)
    df = df.dropna().reset_index(drop=True)

    # Train / Validation split
    train, valid = train_valid_split_time(df, test_size=args.test_size)

    X_cols = [c for c in df.columns if c not in [args.date_col, args.target]]
    y_col = args.target

    # Pipeline
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))
        ]), X_cols)
    ])

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    pipe = Pipeline([
        ('pre', pre),
        ('model', model)
    ])

    Xtr, ytr = train[X_cols], train[y_col]
    Xva, yva = valid[X_cols], valid[y_col]

    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xva)

    # Metrics
    mae = mean_absolute_error(yva, preds)
    
    # ✅ MANUAL RMSE (Compatible with all sklearn versions)
    rmse = (mean_squared_error(yva, preds)) ** 0.5
    
    r2 = r2_score(yva, preds)

    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'n_train': int(len(train)),
        'n_valid': int(len(valid)),
        'horizon': int(args.horizon)
    }

    # Save outputs
    (outdir/'metrics.json').write_text(json.dumps(metrics, indent=2))
    dump(pipe, outdir/'model.pkl')

    # Plot
    plt.figure()
    plt.plot(valid.index, yva, label="Actual")
    plt.plot(valid.index, preds, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted PM2.5")
    plt.savefig(outdir/'actual_vs_pred.png', dpi=150)
    plt.close()

    print("✅ Training complete.")
    print("Saved:", outdir/'model.pkl')
    print("Saved:", outdir/'metrics.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/sample_delhi_aqi.csv')
    parser.add_argument('--target', type=str, default='pm25')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--max_lag', type=int, default=14)
    parser.add_argument('--date_col', type=str, default='date')
    parser.add_argument('--output', type=str, default='outputs')

    args = parser.parse_args()
    main(args)
