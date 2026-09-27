"""Battery lifetime EDA, grouped validation, feature search and held-out evaluation."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from feature_builder import FeatureBuilder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "battery_results"
OUT.mkdir(exist_ok=True)
RAW = pd.read_csv(ROOT / "upload" / "feature_all.csv")
Y = RAW["Lifetime"].to_numpy()
G = RAW["Group"].to_numpy()
DROP = ["Group", "Cell", "Lifetime"]
X = RAW.drop(columns=DROP)


def metric(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    tr, te = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state=42).split(X, Y, G))
    cv = GroupKFold(n_splits=4)
    variants = ["operational", "lean", "raw", "engineered"]
    candidates = {
        "ridge": (Pipeline([("features", FeatureBuilder()), ("impute", SimpleImputer(add_indicator=True)),
                            ("scale", StandardScaler()), ("model", Ridge())]),
                  {"model__alpha": [.1, 10, 100, 1000]}),
        "svr": (Pipeline([("features", FeatureBuilder()), ("impute", SimpleImputer(add_indicator=True)),
                          ("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
                {"model__C": [1, 10, 100], "model__epsilon": [.1, 1], "model__gamma": ["scale", .01]}),
        "extra_trees": (Pipeline([("features", FeatureBuilder()), ("impute", SimpleImputer(add_indicator=True)),
                                  ("model", ExtraTreesRegressor(n_estimators=160, random_state=42, n_jobs=1))]),
                        {"model__min_samples_leaf": [1, 3, 6], "model__max_features": [.7, 1.0]}),
        "hist_gb": (Pipeline([("features", FeatureBuilder()), ("impute", SimpleImputer(add_indicator=True)),
                              ("model", HistGradientBoostingRegressor(random_state=42, early_stopping=False))]),
                    {"model__max_iter": [100, 250], "model__learning_rate": [.03, .08],
                     "model__min_samples_leaf": [5, 12], "model__l2_regularization": [0, 1]}),
    }
    rows, fitted = [], {}
    for variant in variants:
        for name, (pipe, grid) in candidates.items():
            pipe.set_params(features__variant=variant)
            search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv,
                                  n_jobs=2, refit=True, error_score="raise")
            search.fit(X.iloc[tr], Y[tr], groups=G[tr])
            rows.append({"variant": variant, "model": name,
                         "group_cv_MAE": -search.best_score_,
                         "params": json.dumps(search.best_params_, ensure_ascii=False)})
            fitted[(variant, name)] = search.best_estimator_
            print(variant, name, round(-search.best_score_, 3), flush=True)
    ranking = pd.DataFrame(rows).sort_values("group_cv_MAE")
    ranking.to_csv(OUT / "model_search.csv", index=False)
    key = (ranking.iloc[0]["variant"], ranking.iloc[0]["model"])
    best = fitted[key]
    pred = best.predict(X.iloc[te])
    baseline = DummyRegressor(strategy="median").fit(X.iloc[tr], Y[tr])
    basepred = baseline.predict(X.iloc[te])
    results = {"train_n": len(tr), "test_n": len(te),
               "train_groups": int(len(np.unique(G[tr]))), "test_groups": int(len(np.unique(G[te]))),
               "test_groups_names": list(map(str, np.unique(G[te]))),
               "target_train": {"min":float(min(Y[tr])), "max":float(max(Y[tr])), "mean":float(np.mean(Y[tr]))},
               "target_test": {"min":float(min(Y[te])), "max":float(max(Y[te])), "mean":float(np.mean(Y[te]))},
               "best": {"variant":key[0], "model":key[1], "params":best.get_params()["model"].get_params()},
               "test": metric(Y[te], pred), "baseline_test":metric(Y[te], basepred)}
    # Keep scores of untuned preselected alternatives for a transparent diagnostic,
    # without using the holdout to choose or refit the final model.
    oof = cross_val_predict(best, X.iloc[tr], Y[tr], groups=G[tr], cv=cv, n_jobs=2)
    results["train_group_oof"] = metric(Y[tr], oof)
    results["train_random_cell_cv_MAE"] = float(-np.mean(__import__("sklearn.model_selection", fromlist=["cross_val_score"]).cross_val_score(
        best, X.iloc[tr], Y[tr], cv=KFold(4, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error", n_jobs=2)))
    perm = permutation_importance(best, X.iloc[te], Y[te], scoring="neg_mean_absolute_error",
                                  n_repeats=20, random_state=42, n_jobs=2)
    pd.DataFrame({"feature":X.columns, "importance_MAE":perm.importances_mean,
                  "std":perm.importances_std}).sort_values("importance_MAE",ascending=False).to_csv(
                      OUT / "holdout_permutation_importance.csv",index=False)
    pd.DataFrame({"Cell":RAW.iloc[te]["Cell"].to_numpy(), "Group":G[te],
                  "actual":Y[te], "predicted":pred,
                  "absolute_error":np.abs(Y[te]-pred)}).to_csv(OUT / "holdout_predictions.csv",index=False)
    joblib.dump(best, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,default=str))
    print(json.dumps({k:v for k,v in results.items() if k != "best"},indent=2,ensure_ascii=False),flush=True)
