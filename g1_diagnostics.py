"""Reproduce descriptive diagnostics for the held-out G1 group.

This script does not refit or select models using the holdout data.
Run battery_lifetime_analysis.py first to create the fitted pipeline and predictions.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "battery_results"
df = pd.read_csv(ROOT / "upload" / "feature_all.csv")
pred = pd.read_csv(OUT / "holdout_predictions.csv")
model = joblib.load(OUT / "model.joblib")

train_idx, test_idx = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state=42)
                           .split(df, df["Lifetime"], df["Group"]))
assert set(df.iloc[train_idx].Group).isdisjoint(df.iloc[test_idx].Group)
features = model.named_steps["features"].transform(df.drop(columns=["Group", "Cell", "Lifetime"]))
scaled = model.named_steps["scale"].transform(model.named_steps["impute"].transform(features))
g1_idx = df.index[df.Group.eq("G1")].to_numpy()
distance = cdist(scaled[g1_idx], scaled[train_idx])
nearest = train_idx[distance.argmin(axis=1)]

table = (df.loc[g1_idx, ["Cell", "Group", "Chg C-rate", "Dchg C-rate", "DoD",
                         "Lifetime", "capacity_fade_3_0", "mean_deltaQ_dchg_3_0",
                         "mean_dqdv_dchg_mid_3_0"]]
         .merge(pred[["Cell", "predicted", "absolute_error"]], on="Cell"))
table["closest_train_cell"] = df.loc[nearest, "Cell"].to_numpy()
table["closest_train_lifetime"] = df.loc[nearest, "Lifetime"].to_numpy()
table["closest_distance_standardized"] = distance.min(axis=1)
table.to_csv(OUT / "g1_diagnostics.csv", index=False)

heldout_distance = cdist(scaled[test_idx], scaled[train_idx]).min(axis=1)
pd.DataFrame({"Cell": df.iloc[test_idx].Cell.to_numpy(),
              "Group": df.iloc[test_idx].Group.to_numpy(),
              "closest_distance_standardized": heldout_distance}).to_csv(
                  OUT / "holdout_nearest_distance.csv", index=False)

train = features.iloc[train_idx]
g1 = features.iloc[g1_idx]
ranges = pd.DataFrame({"feature":features.columns,
                       "train_min":train.min().to_numpy(),
                       "train_max":train.max().to_numpy(),
                       "g1_min":g1.min().to_numpy(),
                       "g1_max":g1.max().to_numpy()})
ranges["all_G1_below_train_min"] = ranges.g1_max < ranges.train_min
ranges["all_G1_above_train_max"] = ranges.g1_min > ranges.train_max
ranges.to_csv(OUT / "g1_feature_ranges.csv", index=False)

print(table.round(4).to_string(index=False))
print("\nFeatures entirely outside the training range:")
print(ranges.loc[ranges.all_G1_below_train_min | ranges.all_G1_above_train_max]
      .round(5).to_string(index=False))
