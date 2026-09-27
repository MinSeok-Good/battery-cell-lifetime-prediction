"""Feature definitions used by the battery lifetime pipeline."""
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, variant="raw"):
        self.variant = variant

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = X.copy()
        core = ["Chg C-rate", "Dchg C-rate", "DoD", "Q_initial",
                "CV_time_0", "CV_time_3", "capacity_fade_3_0",
                "mean_deltaQ_dchg_3_0", "mean_dqdv_dchg_mid_3_0",
                "avg_stress"]
        if self.variant == "lean":
            return a[core]
        if self.variant == "operational":
            return a[["Chg C-rate", "Dchg C-rate", "DoD", "Q_initial",
                      "Q_ini_V_low", "Q_ini_V_mid", "Q_ini_V_high",
                      "CV_time_0", "chg_stress", "dchg_stress", "avg_stress"]]
        if self.variant == "engineered":
            eps = 1e-6
            a["fade_per_DoD"] = a["capacity_fade_3_0"] / a["DoD"].clip(lower=.05)
            a["cv_time_ratio"] = a["CV_time_3"] / a["CV_time_0"].clip(lower=eps)
            a["charge_discharge_rate_ratio"] = a["Chg C-rate"] / a["Dchg C-rate"].clip(lower=.05)
            a["rate_sum"] = a["Chg C-rate"] + a["Dchg C-rate"]
            a["rate_times_DoD"] = a["rate_sum"] * a["DoD"]
            a["Q_low_mid_gap"] = a["Q_ini_V_mid"] - a["Q_ini_V_low"]
            a["Q_mid_high_gap"] = a["Q_ini_V_high"] - a["Q_ini_V_mid"]
            a["DVA_abs_sum"] = a[[f"delta_Q_DVA{i}" for i in range(1, 5)]].abs().sum(axis=1)
            a["dqdv_mid_minus_low"] = a["mean_dqdv_dchg_mid_3_0"] - a["mean_dqdv_dchg_low_3_0"]
            a["dqdv_high_minus_mid"] = a["mean_dqdv_dchg_high_3_0"] - a["mean_dqdv_dchg_mid_3_0"]
            a["stress_gap"] = a["chg_stress"] - a["dchg_stress"]
            return a
        return a
