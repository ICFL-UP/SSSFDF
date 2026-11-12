#!/usr/bin/env python3
"""
Q-TeleGuard
===========
A strictly tabular Q-learning framework for telecom fraud detection from Excel CDR exports.

Fraud actions:
    0 = LEGITIMATE
    1 = SUBSCRIPTION_FRAUD
    2 = SIM_SWAP_FRAUD

Important:
- If a real fraud label column is available, pass --label-column.
- If no label column is provided, Q-TeleGuard creates heuristic pseudo-labels from CDR behavior.
  In that mode, reported classification metrics measure agreement with the heuristic labels,
  not verified real-world fraud performance.

Example:
    python q_teleguard.py --input cdrs.xlsx --output-dir qteleguard_output
    python q_teleguard.py --input cdrs.xlsx --label-column FRAUD_LABEL

Expected label values when --label-column is used can be:
    0 / legitimate / legit / normal
    1 / subscription_fraud / subscription fraud
    2 / sim_swap_fraud / sim swap fraud
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier


ACTIONS = {
    0: "LEGITIMATE",
    1: "SUBSCRIPTION_FRAUD",
    2: "SIM_SWAP_FRAUD",
}
ACTION_TO_ID = {v: k for k, v in ACTIONS.items()}


# XGBoost benchmark features.
# These are engineered from the same CDR records used by Q-TeleGuard.
XGBOOST_FEATURES = [
    "SUBSCRIPTION_AGE_DAYS",
    "SIM_AGE_DAYS",
    "IDENTITY_CHANGE_COUNT",
    "RADIO_CHANGE_COUNT",
    "AMOUNT_CHARGE_FEE",
    "DATA_VOLUME_MB",
    "CALL_DURATION_SEC",
    "EVENTS_1H",
    "EVENTS_6H",
    "UNIQUE_DEST_6H",
    "DISTANCE_FROM_PREV_KM",
    "GEO_VELOCITY_KMH",
    "IS_NIGHT",
    "IS_WEEKEND",
    "IMSI_CHANGED",
    "IMEI_CHANGED",
    "DEVICE_NAME_CHANGED",
    "DEVICE_BRAND_CHANGED",
    "TOWER_ID_CHANGED",
    "LAC_CHANGED",
    "CELL_ID_CHANGED",
]


REQUIRED_MINIMUM = [
    "CALLING_MSISDN",
    "CALL_START_DATE",
]

DATE_COLUMNS = [
    "CALL_START_DATE",
    "CALL_END_DATE",
    "SUBSCRIPTION_DATE",
    "LAST_SIM_CHANGE_DATE",
]

NUMERIC_COLUMNS = [
    "CALL_DURATION_SEC",
    "DATA_VOLUME_MB",
    "AMOUNT_CHARGE_FEE",
    "LATITUDE",
    "LONGITUDE",
]


@dataclass
class QConfig:
    episodes: int = 60
    alpha: float = 0.15
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.15
    epsilon_decay: float = 0.97
    train_fraction: float = 0.30
    seed: int = 42

    # Fraud timing assumptions used only for state construction / pseudo-labels.
    recent_subscription_days: int = 30
    very_recent_subscription_days: int = 7
    recent_sim_swap_days: int = 7

    # Reward function.
    reward_correct_legit: float = 1.0
    reward_correct_subscription: float = 8.0
    reward_correct_sim_swap: float = 10.0
    penalty_false_positive: float = -3.0
    penalty_missed_subscription: float = -12.0
    penalty_missed_sim_swap: float = -15.0
    penalty_wrong_fraud_type: float = -6.0

    # Small risk-aligned shaping term. It never replaces the classification reward.
    shaping_weight: float = 0.75


class QTeleGuard:
    """
    Tabular Q-learning agent over chronologically ordered CDR events.

    Each CALLING_MSISDN is treated as an episode trajectory. The state is a compact,
    discretized representation of subscriber age, SIM age, recent identity/device/tower
    changes, usage/charge levels, burst activity, and geovelocity.
    """

    def __init__(self, config: QConfig):
        self.cfg = config
        random.seed(config.seed)
        np.random.seed(config.seed)

        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(len(ACTIONS), dtype=np.float64)
        )
        self.bin_edges: Dict[str, List[float]] = {}
        self.label_source = "UNKNOWN"
        self.training_history = {
            "episode": [],
            "training_accuracy": [],
            "mean_reward": [],
            "mean_abs_td_error": [],
            "max_q_delta": [],
            "epsilon": [],
            # State-space diagnostics. These separate states encountered in the
            # current episode from states discovered for the first time.
            "visited_states": [],
            "new_states": [],
            "cumulative_visited_states": [],
            "state_coverage_percent": [],
        }

        # Counts how often each discretized Q-state is encountered across the
        # complete training run. This is used to show which behavioural states
        # dominate training instead of relying only on total Q-table size.
        self.state_visit_counts: Dict[Tuple[int, ...], int] = defaultdict(int)

    # -------------------------------------------------------------------------
    # Data loading and feature engineering
    # -------------------------------------------------------------------------
    @staticmethod
    def load_excel(path: str | Path, sheet_name=0) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        df.columns = [str(c).strip().upper() for c in df.columns]
        missing = [c for c in REQUIRED_MINIMUM if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        return df

    @staticmethod
    def _safe_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    @staticmethod
    def _build_prediction_output_base(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a memory-efficient prediction export.

        The engineered training/test DataFrame can contain many columns and can be
        very large. Copying the whole frame and then asking openpyxl to construct
        an XLSX workbook can require several times the DataFrame size in RAM.
        Keep the most useful identifiers, timestamps, labels and core CDR fields
        in the prediction export instead. The model evaluation itself is unchanged.
        """
        preferred_columns = [
            "CALLING_MSISDN",
            "CALLED_MSISDN",
            "DIALED_NUMBER",
            "CALL_START_DATE",
            "CALL_END_DATE",
            "USAGE_TYPE",
            "ACTIVITY_TYPE",
            "IMSI",
            "IMEI",
            "DEVICE_NAME",
            "DEVICE_BRAND",
            "TOWER_ID",
            "TOWER_NAME",
            "LAC",
            "CELL_ID",
            "CALL_DURATION_SEC",
            "DATA_VOLUME_MB",
            "AMOUNT_CHARGE_FEE",
            "SUBSCRIPTION_AGE_DAYS",
            "SIM_AGE_DAYS",
            "IDENTITY_CHANGE_COUNT",
            "RADIO_CHANGE_COUNT",
            "EVENTS_1H",
            "EVENTS_6H",
            "UNIQUE_DEST_6H",
            "DISTANCE_FROM_PREV_KM",
            "GEO_VELOCITY_KMH",
            "IS_NIGHT",
            "IS_WEEKEND",
            "TARGET",
        ]
        cols = [c for c in preferred_columns if c in df.columns]
        return df.loc[:, cols].copy()

    @staticmethod
    def _save_prediction_table(
        df: pd.DataFrame,
        out_dir: Path,
        stem: str,
        excel_max_rows: int = 100_000,
    ) -> None:
        """
        Save predictions without allowing Excel export to exhaust system memory.

        CSV is always written because pandas streams it efficiently and it has no
        Excel row limit. XLSX is produced only for reasonably sized outputs.
        This avoids the openpyxl MemoryError seen with large CDR test partitions.
        """
        csv_path = out_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False, chunksize=50_000)
        print(f"Prediction CSV written: {csv_path}")

        if len(df) > excel_max_rows:
            print(
                f"Skipping {stem}.xlsx because it contains {len(df):,} rows. "
                f"The complete predictions are available in {csv_path.name}."
            )
            return

        xlsx_path = out_dir / f"{stem}.xlsx"
        try:
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
            print(f"Prediction Excel written: {xlsx_path}")
        except MemoryError:
            # The CSV already contains the full result. Remove a partial workbook
            # if openpyxl managed to create one before memory was exhausted.
            try:
                if xlsx_path.exists():
                    xlsx_path.unlink()
            except OSError:
                pass
            print(
                f"Excel export for {stem} was skipped because available memory "
                f"was insufficient. Use {csv_path.name}, which contains the full output."
            )

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        """Vector-friendly great-circle distance."""
        R = 6371.0088
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def engineer_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()

        for c in DATE_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        for c in NUMERIC_COLUMNS:
            if c in df.columns:
                df[c] = self._safe_numeric(df[c])
            else:
                df[c] = 0.0

        # Ensure optional identity/location columns exist.
        for c in [
            "IMSI", "IMEI", "DEVICE_NAME", "DEVICE_BRAND", "TOWER_ID", "TOWER_NAME",
            "LAC", "CELL_ID", "USAGE_TYPE", "ACTIVITY_TYPE", "CALLED_MSISDN",
            "DIALED_NUMBER", "ENVIRONMENT", "TOWER_TYPE"
        ]:
            if c not in df.columns:
                df[c] = ""

        if "SUBSCRIPTION_DATE" not in df.columns:
            df["SUBSCRIPTION_DATE"] = pd.NaT
        if "LAST_SIM_CHANGE_DATE" not in df.columns:
            df["LAST_SIM_CHANGE_DATE"] = pd.NaT

        df = df.dropna(subset=["CALL_START_DATE", "CALLING_MSISDN"]).copy()
        df["CALLING_MSISDN"] = df["CALLING_MSISDN"].astype(str)
        df.sort_values(["CALLING_MSISDN", "CALL_START_DATE"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Subscriber age and SIM age.
        df["SUBSCRIPTION_AGE_DAYS"] = (
            (df["CALL_START_DATE"] - df["SUBSCRIPTION_DATE"]).dt.total_seconds() / 86400.0
        )
        df["SIM_AGE_DAYS"] = (
            (df["CALL_START_DATE"] - df["LAST_SIM_CHANGE_DATE"]).dt.total_seconds() / 86400.0
        )
        df["SUBSCRIPTION_AGE_DAYS"] = df["SUBSCRIPTION_AGE_DAYS"].fillna(9999).clip(lower=0)
        df["SIM_AGE_DAYS"] = df["SIM_AGE_DAYS"].fillna(9999).clip(lower=0)

        g = df.groupby("CALLING_MSISDN", sort=False)

        # Previous identity / device / tower.
        for c in ["IMSI", "IMEI", "DEVICE_NAME", "DEVICE_BRAND", "TOWER_ID", "LAC", "CELL_ID"]:
            prev = g[c].shift(1)
            cur = df[c].astype(str)
            df[f"{c}_CHANGED"] = ((prev.notna()) & (cur != prev.astype(str))).astype(int)

        prev_time = g["CALL_START_DATE"].shift(1)
        df["SECONDS_SINCE_PREV_EVENT"] = (
            (df["CALL_START_DATE"] - prev_time).dt.total_seconds()
        ).fillna(1e9).clip(lower=1)

        # Geovelocity.
        prev_lat = g["LATITUDE"].shift(1)
        prev_lon = g["LONGITUDE"].shift(1)
        valid_geo = (
            df["LATITUDE"].between(-90, 90)
            & df["LONGITUDE"].between(-180, 180)
            & prev_lat.between(-90, 90)
            & prev_lon.between(-180, 180)
        )
        distance = np.zeros(len(df), dtype=float)
        if valid_geo.any():
            idx = valid_geo.to_numpy()
            distance[idx] = self._haversine_km(
                prev_lat[idx].to_numpy(),
                prev_lon[idx].to_numpy(),
                df.loc[idx, "LATITUDE"].to_numpy(),
                df.loc[idx, "LONGITUDE"].to_numpy(),
            )
        df["DISTANCE_FROM_PREV_KM"] = distance
        hours = df["SECONDS_SINCE_PREV_EVENT"] / 3600.0
        df["GEO_VELOCITY_KMH"] = (df["DISTANCE_FROM_PREV_KM"] / hours).replace(
            [np.inf, -np.inf], 0
        ).fillna(0).clip(0, 2000)

        # Time-based behavior.
        df["HOUR"] = df["CALL_START_DATE"].dt.hour
        df["IS_NIGHT"] = df["HOUR"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        df["IS_WEEKEND"] = (df["CALL_START_DATE"].dt.dayofweek >= 5).astype(int)

        # Rolling event intensity over the previous 1h and 6h per subscriber.
        # Uses chronological history only.
        df["EVENTS_1H"] = 1
        df["EVENTS_6H"] = 1
        df["UNIQUE_DEST_6H"] = 1
        dest_col = "CALLED_MSISDN" if "CALLED_MSISDN" in df.columns else "DIALED_NUMBER"

        for _, idxs in g.indices.items():
            idxs = np.asarray(idxs)
            times = df.loc[idxs, "CALL_START_DATE"].astype("int64").to_numpy() // 10**9
            dests = df.loc[idxs, dest_col].astype(str).to_numpy()

            left1 = left6 = 0
            for pos in range(len(idxs)):
                t = times[pos]
                while times[left1] < t - 3600:
                    left1 += 1
                while times[left6] < t - 21600:
                    left6 += 1

                i = idxs[pos]
                df.at[i, "EVENTS_1H"] = pos - left1 + 1
                df.at[i, "EVENTS_6H"] = pos - left6 + 1
                df.at[i, "UNIQUE_DEST_6H"] = len(set(dests[left6:pos + 1]))

        # Combined identity-change indicators useful for SIM-swap patterns.
        df["IDENTITY_CHANGE_COUNT"] = (
            df["IMSI_CHANGED"] + df["IMEI_CHANGED"] + df["DEVICE_NAME_CHANGED"]
            + df["DEVICE_BRAND_CHANGED"]
        )
        df["RADIO_CHANGE_COUNT"] = (
            df["TOWER_ID_CHANGED"] + df["LAC_CHANGED"] + df["CELL_ID_CHANGED"]
        )

        return df

    # -------------------------------------------------------------------------
    # Label handling
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_label(value) -> int:
        if pd.isna(value):
            return 0
        if isinstance(value, (int, np.integer)) and int(value) in ACTIONS:
            return int(value)
        if isinstance(value, float) and value.is_integer() and int(value) in ACTIONS:
            return int(value)

        s = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        mapping = {
            "0": 0, "legit": 0, "legitimate": 0, "normal": 0, "non_fraud": 0,
            "not_fraud": 0, "genuine": 0,
            "1": 1, "subscription_fraud": 1, "subscription": 1,
            "application_fraud": 1, "new_subscription_fraud": 1,
            "2": 2, "sim_swap": 2, "sim_swap_fraud": 2, "simswap": 2,
        }
        if s not in mapping:
            raise ValueError(
                f"Unrecognized fraud label value {value!r}. "
                f"Use labels 0/1/2 or recognizable names."
            )
        return mapping[s]

    def _fit_pseudo_thresholds(self, train_df: pd.DataFrame):
        def q(col, p, default):
            if col not in train_df or train_df[col].dropna().empty:
                return default
            val = float(train_df[col].quantile(p))
            return default if not np.isfinite(val) else val

        self.pseudo_thresholds = {
            "charge_p90": max(q("AMOUNT_CHARGE_FEE", 0.90, 1.0), 1e-9),
            "data_p90": max(q("DATA_VOLUME_MB", 0.90, 1.0), 1e-9),
            "duration_p90": max(q("CALL_DURATION_SEC", 0.90, 60.0), 1e-9),
            "events_1h_p90": max(q("EVENTS_1H", 0.90, 5.0), 1.0),
            "unique_dest_6h_p90": max(q("UNIQUE_DEST_6H", 0.90, 5.0), 1.0),
        }

    def create_pseudo_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Conservative domain-informed pseudo-labels.

        Subscription fraud indicators:
        - new subscription
        - abnormally high charge / duration / data
        - burst activity or many destinations
        - frequent location/radio changes

        SIM-swap indicators:
        - recent LAST_SIM_CHANGE_DATE
        - IMSI/IMEI/device identity change
        - unusual activity after swap
        - high geovelocity or rapid tower movement
        """
        t = self.pseudo_thresholds

        new_sub = df["SUBSCRIPTION_AGE_DAYS"] <= self.cfg.recent_subscription_days
        very_new_sub = df["SUBSCRIPTION_AGE_DAYS"] <= self.cfg.very_recent_subscription_days
        recent_swap = df["SIM_AGE_DAYS"] <= self.cfg.recent_sim_swap_days

        high_charge = df["AMOUNT_CHARGE_FEE"] >= t["charge_p90"]
        high_data = df["DATA_VOLUME_MB"] >= t["data_p90"]
        high_duration = df["CALL_DURATION_SEC"] >= t["duration_p90"]
        burst = df["EVENTS_1H"] >= t["events_1h_p90"]
        many_dest = df["UNIQUE_DEST_6H"] >= t["unique_dest_6h_p90"]
        identity_change = df["IDENTITY_CHANGE_COUNT"] >= 1
        strong_identity_change = df["IDENTITY_CHANGE_COUNT"] >= 2
        suspicious_velocity = df["GEO_VELOCITY_KMH"] >= 500
        radio_churn = df["RADIO_CHANGE_COUNT"] >= 2

        sub_score = (
            2.0 * new_sub.astype(float)
            + 1.0 * very_new_sub.astype(float)
            + 1.4 * high_charge.astype(float)
            + 1.0 * high_data.astype(float)
            + 0.8 * high_duration.astype(float)
            + 1.0 * burst.astype(float)
            + 1.1 * many_dest.astype(float)
            + 0.5 * radio_churn.astype(float)
        )

        sim_score = (
            3.0 * recent_swap.astype(float)
            + 1.7 * identity_change.astype(float)
            + 1.2 * strong_identity_change.astype(float)
            + 1.2 * high_charge.astype(float)
            + 0.8 * high_data.astype(float)
            + 1.0 * burst.astype(float)
            + 1.6 * suspicious_velocity.astype(float)
        )

        labels = np.zeros(len(df), dtype=int)
        sim_mask = (sim_score >= 5.0) & recent_swap
        sub_mask = (sub_score >= 4.8) & new_sub

        labels[sub_mask] = 1
        labels[sim_mask] = 2

        # If both fire, prefer the stronger normalized score.
        both = sim_mask & sub_mask
        labels[both & (sub_score > sim_score)] = 1
        labels[both & (sim_score >= sub_score)] = 2

        return pd.Series(labels, index=df.index, name="TARGET")

    # -------------------------------------------------------------------------
    # State space
    # -------------------------------------------------------------------------
    @staticmethod
    def _quantile_edges(s: pd.Series, qs=(0.50, 0.80, 0.95)) -> List[float]:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return [0.0, 1.0, 2.0]
        raw = [float(vals.quantile(q)) for q in qs]
        edges = []
        for x in raw:
            if not np.isfinite(x):
                continue
            if not edges or x > edges[-1]:
                edges.append(x)
        if not edges:
            edges = [0.0]
        return edges

    def fit_state_bins(self, train_df: pd.DataFrame):
        self.bin_edges = {
            "CHARGE": self._quantile_edges(train_df["AMOUNT_CHARGE_FEE"]),
            "DATA": self._quantile_edges(train_df["DATA_VOLUME_MB"]),
            "DURATION": self._quantile_edges(train_df["CALL_DURATION_SEC"]),
            "EVENTS_1H": self._quantile_edges(train_df["EVENTS_1H"]),
            "UNIQUE_DEST_6H": self._quantile_edges(train_df["UNIQUE_DEST_6H"]),
            "VELOCITY": [50.0, 150.0, 500.0],
        }

    @staticmethod
    def _bin(v: float, edges: List[float]) -> int:
        return int(np.digitize([float(v)], edges, right=False)[0])

    @staticmethod
    def _age_bin(days: float, cutoffs=(7, 30, 90)) -> int:
        return int(np.digitize([float(days)], cutoffs, right=True)[0])

    def state(self, row: pd.Series) -> Tuple[int, ...]:
        """
        Compact tabular state. Deliberately avoids raw MSISDN/IMSI/IMEI values so
        the Q-table learns behavioral patterns rather than memorizing identities.
        """
        return (
            self._age_bin(row["SUBSCRIPTION_AGE_DAYS"]),
            self._age_bin(row["SIM_AGE_DAYS"]),
            min(int(row["IDENTITY_CHANGE_COUNT"]), 2),
            min(int(row["RADIO_CHANGE_COUNT"]), 2),
            self._bin(row["AMOUNT_CHARGE_FEE"], self.bin_edges["CHARGE"]),
            self._bin(row["DATA_VOLUME_MB"], self.bin_edges["DATA"]),
            self._bin(row["CALL_DURATION_SEC"], self.bin_edges["DURATION"]),
            self._bin(row["EVENTS_1H"], self.bin_edges["EVENTS_1H"]),
            self._bin(row["UNIQUE_DEST_6H"], self.bin_edges["UNIQUE_DEST_6H"]),
            self._bin(row["GEO_VELOCITY_KMH"], self.bin_edges["VELOCITY"]),
            int(row["IS_NIGHT"]),
        )

    # -------------------------------------------------------------------------
    # Reward function
    # -------------------------------------------------------------------------
    def risk_scores(self, row: pd.Series) -> Tuple[float, float]:
        """Returns bounded subscription-fraud and SIM-swap risk scores in [0, 1]."""
        sub = 0.0
        sim = 0.0

        if row["SUBSCRIPTION_AGE_DAYS"] <= 30:
            sub += 0.30
        if row["SUBSCRIPTION_AGE_DAYS"] <= 7:
            sub += 0.15
        if self._bin(row["AMOUNT_CHARGE_FEE"], self.bin_edges["CHARGE"]) >= 3:
            sub += 0.20
        if self._bin(row["EVENTS_1H"], self.bin_edges["EVENTS_1H"]) >= 3:
            sub += 0.15
        if self._bin(row["UNIQUE_DEST_6H"], self.bin_edges["UNIQUE_DEST_6H"]) >= 3:
            sub += 0.15
        if row["RADIO_CHANGE_COUNT"] >= 2:
            sub += 0.05

        if row["SIM_AGE_DAYS"] <= 7:
            sim += 0.40
        if row["IDENTITY_CHANGE_COUNT"] >= 1:
            sim += 0.25
        if row["IDENTITY_CHANGE_COUNT"] >= 2:
            sim += 0.10
        if row["GEO_VELOCITY_KMH"] >= 500:
            sim += 0.15
        if self._bin(row["AMOUNT_CHARGE_FEE"], self.bin_edges["CHARGE"]) >= 3:
            sim += 0.10

        return min(sub, 1.0), min(sim, 1.0)

    def reward(self, action: int, true_label: int, row: pd.Series) -> float:
        """
        Structured asymmetric reward.

        Fraud misses are penalized more heavily than false positives because the
        operational cost of allowing fraudulent telecom activity can be high.
        SIM-swap misses receive the strongest penalty.

        A small shaping term rewards alignment between the selected fraud action
        and observed risk features. The class-correctness reward remains dominant.
        """
        if action == true_label:
            if true_label == 0:
                base = self.cfg.reward_correct_legit
            elif true_label == 1:
                base = self.cfg.reward_correct_subscription
            else:
                base = self.cfg.reward_correct_sim_swap
        else:
            if true_label == 1 and action == 0:
                base = self.cfg.penalty_missed_subscription
            elif true_label == 2 and action == 0:
                base = self.cfg.penalty_missed_sim_swap
            elif true_label == 0 and action in (1, 2):
                base = self.cfg.penalty_false_positive
            else:
                base = self.cfg.penalty_wrong_fraud_type

        sub_risk, sim_risk = self.risk_scores(row)
        if action == 1:
            shaping = sub_risk - 0.5
        elif action == 2:
            shaping = sim_risk - 0.5
        else:
            shaping = 0.5 - max(sub_risk, sim_risk)

        return float(base + self.cfg.shaping_weight * shaping)

    # -------------------------------------------------------------------------
    # Q-learning
    # -------------------------------------------------------------------------
    def choose_action(self, s: Tuple[int, ...], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        q = self.q_table[s]
        max_q = q.max()
        ties = np.flatnonzero(np.isclose(q, max_q))
        return int(np.random.choice(ties))

    def train(self, train_df: pd.DataFrame):
        subscriber_groups = [
            grp.index.to_numpy()
            for _, grp in train_df.groupby("CALLING_MSISDN", sort=False)
        ]

        epsilon = self.cfg.epsilon_start

        # Track state discovery independently of Q-table updates.  The previous
        # implementation recorded len(self.q_table), which usually became fixed
        # after the first full pass through the training CDRs and therefore
        # produced a flat state-space graph.
        global_visited_states = set()

        for ep in range(1, self.cfg.episodes + 1):
            random.shuffle(subscriber_groups)
            total_reward = 0.0
            td_errors = []
            max_delta = 0.0
            steps = 0
            correct = 0

            episode_states = set()
            new_states_this_episode = 0

            for idxs in subscriber_groups:
                if len(idxs) == 0:
                    continue

                for pos, idx in enumerate(idxs):
                    row = train_df.loc[idx]
                    s = self.state(row)

                    # State-space exploration diagnostics.
                    episode_states.add(s)
                    self.state_visit_counts[s] += 1
                    if s not in global_visited_states:
                        global_visited_states.add(s)
                        new_states_this_episode += 1

                    a = self.choose_action(s, epsilon)
                    true_label = int(row["TARGET"])
                    r = self.reward(a, true_label, row)
                    if a == true_label:
                        correct += 1

                    terminal = pos == len(idxs) - 1
                    if terminal:
                        bootstrap = 0.0
                    else:
                        next_row = train_df.loc[idxs[pos + 1]]
                        s2 = self.state(next_row)
                        bootstrap = self.cfg.gamma * float(np.max(self.q_table[s2]))

                    old_q = self.q_table[s][a]
                    td_error = r + bootstrap - old_q
                    new_q = old_q + self.cfg.alpha * td_error
                    self.q_table[s][a] = new_q

                    total_reward += r
                    td_errors.append(abs(td_error))
                    max_delta = max(max_delta, abs(new_q - old_q))
                    steps += 1

            mean_reward = total_reward / max(steps, 1)
            mean_td = float(np.mean(td_errors)) if td_errors else 0.0
            training_accuracy = correct / max(steps, 1)

            self.training_history["episode"].append(ep)
            self.training_history["training_accuracy"].append(training_accuracy)
            self.training_history["mean_reward"].append(mean_reward)
            self.training_history["mean_abs_td_error"].append(mean_td)
            self.training_history["max_q_delta"].append(max_delta)
            self.training_history["epsilon"].append(epsilon)
            self.training_history["visited_states"].append(len(episode_states))
            self.training_history["new_states"].append(new_states_this_episode)
            self.training_history["cumulative_visited_states"].append(
                len(global_visited_states)
            )

            # Coverage is normalized after training against the final discovered
            # state space. A temporary value is kept here so all history arrays
            # have matching lengths during training.
            self.training_history["state_coverage_percent"].append(0.0)

            print(
                f"Episode {ep:03d}/{self.cfg.episodes} | "
                f"train accuracy={training_accuracy:7.4f} | "
                f"mean reward={mean_reward:8.4f} | "
                f"mean |TD|={mean_td:8.4f} | "
                f"max ΔQ={max_delta:8.4f} | "
                f"epsilon={epsilon:6.4f} | "
                f"episode states={len(episode_states)} | "
                f"new states={new_states_this_episode} | "
                f"cumulative states={len(global_visited_states)}"
            )

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)

        # Convert episode state counts into coverage of the final discovered
        # training state space. This makes the percentage directly comparable
        # from episode to episode.
        final_state_count = max(len(global_visited_states), 1)
        self.training_history["state_coverage_percent"] = [
            100.0 * n / final_state_count
            for n in self.training_history["visited_states"]
        ]

    def q_scores(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros((len(df), len(ACTIONS)), dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            s = self.state(row)
            scores[i] = self.q_table.get(s, np.zeros(len(ACTIONS)))
        return scores

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(np.clip(x, -50, 50))
        den = ex.sum(axis=1, keepdims=True)
        den[den == 0] = 1.0
        return ex / den

    def predict(self, df: pd.DataFrame):
        q = self.q_scores(df)
        pred = np.argmax(q, axis=1)
        prob = self._softmax(q)
        return pred, prob, q

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame, out_dir: Path) -> dict:
        y_true = test_df["TARGET"].astype(int).to_numpy()
        prediction_start = time.perf_counter()
        y_pred, prob, q_values = self.predict(test_df)
        prediction_time = time.perf_counter() - prediction_start

        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["prediction_latency_seconds"] = float(prediction_time)
        metrics["prediction_latency_ms_per_record"] = float((prediction_time * 1000.0) / max(len(test_df), 1))

        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )
        for i, name in ACTIONS.items():
            metrics[f"{name}_precision"] = float(p[i])
            metrics[f"{name}_recall"] = float(r[i])
            metrics[f"{name}_sensitivity"] = float(r[i])
            metrics[f"{name}_f1"] = float(f1[i])
            metrics[f"{name}_support"] = int(support[i])

        macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["macro_precision"] = float(macro[0])
        metrics["macro_recall"] = float(macro[1])
        metrics["macro_f1"] = float(macro[2])
        metrics["weighted_precision"] = float(weighted[0])
        metrics["weighted_recall"] = float(weighted[1])
        metrics["weighted_f1"] = float(weighted[2])
        metrics["sensitivity"] = float(macro[1])
        metrics["overall_performance"] = float(np.mean([metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"]]))

        # Fraud/no-fraud operational metrics.
        fraud_true = (y_true != 0).astype(int)
        fraud_pred = (y_pred != 0).astype(int)
        fp = int(((fraud_true == 0) & (fraud_pred == 1)).sum())
        fn = int(((fraud_true == 1) & (fraud_pred == 0)).sum())
        tp = int(((fraud_true == 1) & (fraud_pred == 1)).sum())
        tn = int(((fraud_true == 0) & (fraud_pred == 0)).sum())
        metrics["fraud_true_positive"] = tp
        metrics["fraud_false_positive"] = fp
        metrics["fraud_false_negative"] = fn
        metrics["fraud_true_negative"] = tn
        metrics["fraud_detection_rate_recall"] = float(tp / max(tp + fn, 1))
        metrics["false_alarm_rate"] = float(fp / max(fp + tn, 1))
        metrics["miss_rate"] = float(fn / max(tp + fn, 1))

        # Per-class PR-AUC and ROC-AUC from Q-policy scores.
        # PR-AUC is calculated as Average Precision, a standard summary for
        # imbalanced fraud-detection problems.
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        pr_auc_values = []
        roc_auc_values = []
        auc_supports = []

        for i, name in ACTIONS.items():
            y_i = y_bin[:, i]
            class_support_i = int(y_i.sum())
            auc_supports.append(class_support_i)

            if class_support_i > 0:
                pr_auc = float(average_precision_score(y_i, prob[:, i]))
                metrics[f"{name}_pr_auc"] = pr_auc
                pr_auc_values.append(pr_auc)
            else:
                metrics[f"{name}_pr_auc"] = None
                pr_auc_values.append(np.nan)

            if len(np.unique(y_i)) == 2:
                roc_auc = float(roc_auc_score(y_i, prob[:, i]))
                metrics[f"{name}_roc_auc"] = roc_auc
                roc_auc_values.append(roc_auc)
            else:
                metrics[f"{name}_roc_auc"] = None
                roc_auc_values.append(np.nan)

        pr_arr = np.asarray(pr_auc_values, dtype=float)
        roc_arr = np.asarray(roc_auc_values, dtype=float)
        sup_arr = np.asarray(auc_supports, dtype=float)

        valid_pr = np.isfinite(pr_arr)
        if valid_pr.any():
            metrics["macro_pr_auc"] = float(np.nanmean(pr_arr))
            if sup_arr[valid_pr].sum() > 0:
                metrics["weighted_pr_auc"] = float(
                    np.sum(pr_arr[valid_pr] * sup_arr[valid_pr]) /
                    np.sum(sup_arr[valid_pr])
                )

        valid_roc = np.isfinite(roc_arr)
        if valid_roc.any():
            metrics["macro_roc_auc_ovr"] = float(np.nanmean(roc_arr))
            if sup_arr[valid_roc].sum() > 0:
                metrics["weighted_roc_auc_ovr"] = float(
                    np.sum(roc_arr[valid_roc] * sup_arr[valid_roc]) /
                    np.sum(sup_arr[valid_roc])
                )

        # State/Q metrics.
        states = [self.state(row) for _, row in test_df.iterrows()]
        seen = np.array([s in self.q_table for s in states], dtype=bool)
        metrics["test_state_coverage"] = float(seen.mean()) if len(seen) else 0.0
        metrics["q_table_states"] = int(len(self.q_table))
        metrics["q_teleguard_training_accuracy"] = float(
            self.training_history["training_accuracy"][-1]
        )
        metrics["q_teleguard_average_reward"] = float(
            np.mean(self.training_history["mean_reward"])
        )
        metrics["final_mean_reward"] = float(self.training_history["mean_reward"][-1])
        metrics["final_mean_abs_td_error"] = float(
            self.training_history["mean_abs_td_error"][-1]
        )
        metrics["final_max_q_delta"] = float(self.training_history["max_q_delta"][-1])

        # Classification report.
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[ACTIONS[i] for i in range(3)],
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv")

        # Confusion matrix CSV.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(
            cm,
            index=[f"TRUE_{ACTIONS[i]}" for i in range(3)],
            columns=[f"PRED_{ACTIONS[i]}" for i in range(3)],
        )
        cm_df.to_csv(out_dir / "confusion_matrix.csv")

        # Predictions. Build a compact export and save CSV unconditionally.
        # XLSX is created only when the output is small enough to avoid openpyxl
        # exhausting RAM on large CDR test partitions.
        pred_out = self._build_prediction_output_base(test_df)
        pred_out["PREDICTED_ACTION_ID"] = y_pred
        pred_out["PREDICTED_FRAUD_CLASS"] = [ACTIONS[int(x)] for x in y_pred]
        pred_out["Q_LEGITIMATE"] = q_values[:, 0]
        pred_out["Q_SUBSCRIPTION_FRAUD"] = q_values[:, 1]
        pred_out["Q_SIM_SWAP_FRAUD"] = q_values[:, 2]
        pred_out["POLICY_SCORE_LEGITIMATE"] = prob[:, 0]
        pred_out["POLICY_SCORE_SUBSCRIPTION_FRAUD"] = prob[:, 1]
        pred_out["POLICY_SCORE_SIM_SWAP_FRAUD"] = prob[:, 2]
        self._save_prediction_table(
            pred_out, out_dir, "q_teleguard_predictions"
        )
        del pred_out

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

        self.make_plots(test_df, y_true, y_pred, prob, out_dir)
        return metrics

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    def make_plots(self, test_df, y_true, y_pred, prob, out_dir: Path):
        hist = pd.DataFrame(self.training_history)
        hist.to_csv(out_dir / "q_learning_training_history.csv", index=False)

        def save_line(x, y, xlabel, ylabel, title, filename):
            plt.figure(figsize=(8, 5))
            plt.plot(x, y)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=150)
            plt.close()

        save_line(
            hist["episode"], hist["training_accuracy"],
            "Episode", "Training Accuracy",
            "Q-TeleGuard Training Accuracy",
            "01_q_teleguard_training_accuracy.png",
        )
        save_line(
            hist["episode"], hist["mean_reward"],
            "Episode", "Average Reward",
            "Q-TeleGuard Average Reward",
            "02_q_teleguard_average_reward.png",
        )
        save_line(
            hist["episode"], hist["mean_abs_td_error"],
            "Episode", "Mean Absolute TD Error",
            "Q-TeleGuard: TD Error Convergence",
            "03_training_td_error.png",
        )
        save_line(
            hist["episode"], hist["max_q_delta"],
            "Episode", "Maximum |ΔQ|",
            "Q-TeleGuard: Q-Table Update Convergence",
            "04_q_table_convergence.png",
        )
        save_line(
            hist["episode"], hist["epsilon"],
            "Episode", "Epsilon",
            "Q-TeleGuard: Exploration Decay",
            "05_epsilon_decay.png",
        )
        # Improved state-space diagnostics.  The old graph plotted only the
        # total Q-table size, which normally became constant after the first
        # complete pass through the training CDRs.  The replacement separates
        # per-episode coverage, first-time discovery, and cumulative discovery.
        plt.figure(figsize=(10, 6))
        plt.plot(
            hist["episode"],
            hist["visited_states"],
            label="Unique states visited in episode",
        )
        plt.plot(
            hist["episode"],
            hist["new_states"],
            label="New states discovered",
        )
        plt.plot(
            hist["episode"],
            hist["cumulative_visited_states"],
            label="Cumulative unique states",
        )
        plt.xlabel("Episode")
        plt.ylabel("Number of Q States")
        plt.title("Q-TeleGuard: State-Space Exploration During Training")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "06_visited_states.png", dpi=150)
        plt.close()

        save_line(
            hist["episode"], hist["state_coverage_percent"],
            "Episode", "State Coverage (%)",
            "Q-TeleGuard: Episode State Coverage",
            "06b_state_coverage_percent.png",
        )

        # Show which behavioural Q-states dominate training.  This is often more
        # informative than total state count when every episode processes the
        # same training records.
        if self.state_visit_counts:
            top_state_visits = sorted(
                self.state_visit_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:20]
            state_labels = [f"S{i + 1}" for i in range(len(top_state_visits))]
            visit_counts = [count for _, count in top_state_visits]

            plt.figure(figsize=(10, 6))
            plt.bar(np.arange(len(visit_counts)), visit_counts)
            plt.xticks(
                np.arange(len(state_labels)),
                state_labels,
                rotation=45,
                ha="right",
            )
            plt.xlabel("Most Frequently Visited Q States")
            plt.ylabel("Training Visit Count")
            plt.title("Q-TeleGuard: Top 20 Most Frequently Visited States")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / "06c_top_state_visit_frequency.png", dpi=150)
            plt.close()

            # Save the underlying state tuples and counts so the graph remains
            # auditable and the S1..S20 labels can be interpreted.
            state_visit_rows = []
            for rank, (state_tuple, count) in enumerate(top_state_visits, start=1):
                state_visit_rows.append({
                    "state_label": f"S{rank}",
                    "state_tuple": str(state_tuple),
                    "visit_count": int(count),
                })
            pd.DataFrame(state_visit_rows).to_csv(
                out_dir / "top_state_visit_frequency.csv",
                index=False,
            )


        # Q-table heatmap: top learned states by absolute Q magnitude.
        if len(self.q_table) > 0:
            q_items = sorted(
                self.q_table.items(),
                key=lambda kv: float(np.max(np.abs(kv[1]))),
                reverse=True,
            )[:50]

            q_matrix = np.vstack([q for _, q in q_items])
            q_state_labels = [f"S{i+1}" for i in range(len(q_items))]

            plt.figure(figsize=(9, max(5, min(14, len(q_items) * 0.28))))
            plt.imshow(q_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm")
            plt.colorbar(label="Q-value")
            plt.xticks(np.arange(3), [ACTIONS[i] for i in range(3)], rotation=20)
            plt.yticks(np.arange(len(q_state_labels)), q_state_labels)
            plt.xlabel("Action")
            plt.ylabel("Top learned states")
            plt.title("Q-table Heatmap - Q-TeleGuard")
            plt.tight_layout()
            plt.savefig(out_dir / "07_q_table_heatmap.png", dpi=150)
            plt.close()

        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix - Q-TeleGuard")
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, [ACTIONS[i] for i in range(3)], rotation=20)
        plt.yticks(tick_marks, [ACTIONS[i] for i in range(3)])
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(3):
            for j in range(3):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.tight_layout()
        plt.savefig(out_dir / "08_confusion_matrix_q_teleguard.png", dpi=150)
        plt.close()

        # Precision, recall/sensitivity, F1 and support per class.
        p, r, f1, class_support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )
        x = np.arange(3)
        width = 0.20

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, p, width, label="Precision")
        plt.bar(x, r, width, label="Recall / Sensitivity")
        plt.bar(x + width, f1, width, label="F1-score")
        plt.xticks(x, [ACTIONS[i] for i in range(3)], rotation=20)
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Precision, Recall, Sensitivity and F1-score - Q-TeleGuard")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "09_precision_recall_sensitivity_f1.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(x, class_support)
        plt.xticks(x, [ACTIONS[i] for i in range(3)], rotation=20)
        plt.ylabel("Support")
        plt.title("Support per Class - Q-TeleGuard")
        plt.tight_layout()
        plt.savefig(out_dir / "09b_support_per_class.png", dpi=150)
        plt.close()

        # ROC and PR curves where classes are present.
        y_bin = label_binarize(y_true, classes=[0, 1, 2])

        plt.figure(figsize=(8, 6))
        plotted = False
        for i in range(3):
            if len(np.unique(y_bin[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
            plt.plot(fpr, tpr, label=ACTIONS[i])
            plotted = True
        if plotted:
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("One-vs-Rest ROC Curves from Greedy Q Policy Scores")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "10_roc_curves.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plotted = False
        for i in range(3):
            if y_bin[:, i].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_bin[:, i], prob[:, i])
            plt.plot(recall, precision, label=ACTIONS[i])
            plotted = True
        if plotted:
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("One-vs-Rest Precision-Recall Curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "11_precision_recall_curves.png", dpi=150)
        plt.close()

        # PR-AUC scores by class.
        pr_scores = []
        pr_labels = []
        for i in range(3):
            if y_bin[:, i].sum() > 0:
                pr_scores.append(float(average_precision_score(y_bin[:, i], prob[:, i])))
                pr_labels.append(ACTIONS[i])
        if pr_scores:
            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(len(pr_scores)), pr_scores)
            plt.xticks(np.arange(len(pr_scores)), pr_labels, rotation=20)
            plt.ylim(0, 1.05)
            plt.ylabel("PR-AUC / Average Precision")
            plt.title("PR-AUC Scores - Q-TeleGuard")
            plt.tight_layout()
            plt.savefig(out_dir / "12_pr_auc_scores.png", dpi=150)
            plt.close()

        # ROC-AUC scores by class.
        roc_scores = []
        roc_labels = []
        for i in range(3):
            if len(np.unique(y_bin[:, i])) == 2:
                roc_scores.append(float(roc_auc_score(y_bin[:, i], prob[:, i])))
                roc_labels.append(ACTIONS[i])
        if roc_scores:
            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(len(roc_scores)), roc_scores)
            plt.xticks(np.arange(len(roc_scores)), roc_labels, rotation=20)
            plt.ylim(0, 1.05)
            plt.ylabel("ROC-AUC")
            plt.title("ROC-AUC Scores - Q-TeleGuard")
            plt.tight_layout()
            plt.savefig(out_dir / "12b_roc_auc_scores.png", dpi=150)
            plt.close()

        # Fraud rate over time on the test interval.
        timeline = pd.DataFrame({
            "CALL_START_DATE": test_df["CALL_START_DATE"].values,
            "TRUE_FRAUD": (y_true != 0).astype(int),
            "PRED_FRAUD": (y_pred != 0).astype(int),
        }).dropna()
        if not timeline.empty:
            timeline["DAY"] = pd.to_datetime(timeline["CALL_START_DATE"]).dt.floor("D")
            daily = timeline.groupby("DAY")[["TRUE_FRAUD", "PRED_FRAUD"]].mean()
            plt.figure(figsize=(10, 5))
            plt.plot(daily.index, daily["TRUE_FRAUD"], label="True fraud rate")
            plt.plot(daily.index, daily["PRED_FRAUD"], label="Predicted fraud rate")
            plt.xlabel("Date")
            plt.ylabel("Fraud rate")
            plt.title("Daily Fraud Rate: Actual vs Q-Learning Policy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "13_daily_fraud_rate.png", dpi=150)
            plt.close()


    # -------------------------------------------------------------------------
    # XGBoost benchmark
    # -------------------------------------------------------------------------
    def train_xgboost_benchmark(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        q_metrics: dict,
        out_dir: Path,
    ) -> dict:
        """
        Train XGBoost as a separate supervised benchmark.

        Q-TeleGuard remains the reinforcement-learning model. XGBoost is used
        only to provide a conventional supervised-learning comparison on the
        exact same chronological train/test partition.
        """
        feature_cols = [c for c in XGBOOST_FEATURES if c in train_df.columns]
        if not feature_cols:
            raise ValueError("No XGBoost benchmark features were found.")

        X_train = (
            train_df[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        X_test = (
            test_df[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        y_train = train_df["TARGET"].astype(int).to_numpy()
        y_test = test_df["TARGET"].astype(int).to_numpy()

        present_train = np.unique(y_train)
        if len(present_train) < 2:
            raise ValueError(
                "XGBoost benchmark requires at least two fraud classes in the training partition."
            )

        # XGBoost requires contiguous class IDs. A chronological split may
        # contain, for example, classes {0, 2} but no subscription-fraud class 1.
        # Remap only for XGBoost training, then restore the original Q-TeleGuard
        # action IDs for evaluation and comparison.
        original_classes = np.sort(np.unique(y_train))
        class_to_xgb = {int(cls): idx for idx, cls in enumerate(original_classes)}
        xgb_to_class = {idx: int(cls) for idx, cls in enumerate(original_classes)}
        y_train_xgb = np.array([class_to_xgb[int(v)] for v in y_train], dtype=int)

        if len(original_classes) == 2:
            model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=self.cfg.seed,
                n_jobs=-1,
            )
        else:
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(original_classes),
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=1,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                random_state=self.cfg.seed,
                n_jobs=-1,
            )

        xgb_train_start = time.perf_counter()
        model.fit(X_train, y_train_xgb)
        xgb_training_time = time.perf_counter() - xgb_train_start

        xgb_prediction_start = time.perf_counter()
        pred_xgb_ids = model.predict(X_test).astype(int)
        xgb_pred = np.array(
            [xgb_to_class[int(v)] for v in pred_xgb_ids],
            dtype=int,
        )
        raw_prob = model.predict_proba(X_test)
        xgb_prediction_time = time.perf_counter() - xgb_prediction_start

        # Expand probabilities back into the three Q-TeleGuard action columns.
        xgb_prob = np.zeros((len(X_test), 3), dtype=float)
        for prob_col, xgb_class_id in enumerate(model.classes_):
            original_class_id = xgb_to_class[int(xgb_class_id)]
            xgb_prob[:, original_class_id] = raw_prob[:, prob_col]

        # Ensure each row is a valid probability vector.
        row_sum = xgb_prob.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        xgb_prob = xgb_prob / row_sum

        xgb_metrics = {}
        xgb_metrics["accuracy"] = float(accuracy_score(y_test, xgb_pred))
        xgb_metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_test, xgb_pred)
        )
        xgb_metrics["training_time_seconds"] = float(xgb_training_time)
        xgb_metrics["prediction_latency_seconds"] = float(xgb_prediction_time)
        xgb_metrics["prediction_latency_ms_per_record"] = float((xgb_prediction_time * 1000.0) / max(len(X_test), 1))

        p, r, f1, support = precision_recall_fscore_support(
            y_test,
            xgb_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )

        for i, name in ACTIONS.items():
            xgb_metrics[f"{name}_precision"] = float(p[i])
            xgb_metrics[f"{name}_recall"] = float(r[i])
            xgb_metrics[f"{name}_sensitivity"] = float(r[i])
            xgb_metrics[f"{name}_f1"] = float(f1[i])
            xgb_metrics[f"{name}_support"] = int(support[i])

        macro = precision_recall_fscore_support(
            y_test, xgb_pred, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            y_test, xgb_pred, average="weighted", zero_division=0
        )

        xgb_metrics["macro_precision"] = float(macro[0])
        xgb_metrics["macro_recall"] = float(macro[1])
        xgb_metrics["macro_f1"] = float(macro[2])
        xgb_metrics["weighted_precision"] = float(weighted[0])
        xgb_metrics["weighted_recall"] = float(weighted[1])
        xgb_metrics["weighted_f1"] = float(weighted[2])
        xgb_metrics["sensitivity"] = float(macro[1])
        xgb_metrics["overall_performance"] = float(np.mean([xgb_metrics["accuracy"], xgb_metrics["balanced_accuracy"], xgb_metrics["macro_precision"], xgb_metrics["macro_recall"], xgb_metrics["macro_f1"]]))

        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        pr_values = []
        roc_values = []
        auc_supports = []

        for i, name in ACTIONS.items():
            y_i = y_bin[:, i]
            auc_supports.append(int(y_i.sum()))

            if y_i.sum() > 0:
                pr_auc = float(average_precision_score(y_i, xgb_prob[:, i]))
                xgb_metrics[f"{name}_pr_auc"] = pr_auc
                pr_values.append(pr_auc)
            else:
                xgb_metrics[f"{name}_pr_auc"] = None
                pr_values.append(np.nan)

            if len(np.unique(y_i)) == 2:
                roc_auc = float(roc_auc_score(y_i, xgb_prob[:, i]))
                xgb_metrics[f"{name}_roc_auc"] = roc_auc
                roc_values.append(roc_auc)
            else:
                xgb_metrics[f"{name}_roc_auc"] = None
                roc_values.append(np.nan)

        pr_arr = np.asarray(pr_values, dtype=float)
        roc_arr = np.asarray(roc_values, dtype=float)
        sup_arr = np.asarray(auc_supports, dtype=float)

        valid_pr = np.isfinite(pr_arr)
        if valid_pr.any():
            xgb_metrics["macro_pr_auc"] = float(np.nanmean(pr_arr))
            if sup_arr[valid_pr].sum() > 0:
                xgb_metrics["weighted_pr_auc"] = float(
                    np.sum(pr_arr[valid_pr] * sup_arr[valid_pr])
                    / np.sum(sup_arr[valid_pr])
                )

        valid_roc = np.isfinite(roc_arr)
        if valid_roc.any():
            xgb_metrics["macro_roc_auc_ovr"] = float(np.nanmean(roc_arr))
            if sup_arr[valid_roc].sum() > 0:
                xgb_metrics["weighted_roc_auc_ovr"] = float(
                    np.sum(roc_arr[valid_roc] * sup_arr[valid_roc])
                    / np.sum(sup_arr[valid_roc])
                )

        # Save XGBoost predictions beside the Q-TeleGuard outputs using the
        # same memory-safe export strategy.
        xgb_out = self._build_prediction_output_base(test_df)
        xgb_out["XGB_PREDICTED_ACTION_ID"] = xgb_pred
        xgb_out["XGB_PREDICTED_FRAUD_CLASS"] = [
            ACTIONS.get(int(v), str(v)) for v in xgb_pred
        ]
        xgb_out["XGB_PROB_LEGITIMATE"] = xgb_prob[:, 0]
        xgb_out["XGB_PROB_SUBSCRIPTION_FRAUD"] = xgb_prob[:, 1]
        xgb_out["XGB_PROB_SIM_SWAP_FRAUD"] = xgb_prob[:, 2]
        self._save_prediction_table(
            xgb_out, out_dir, "xgboost_predictions"
        )
        del xgb_out

        with open(out_dir / "xgboost_metrics.json", "w", encoding="utf-8") as f:
            json.dump(xgb_metrics, f, indent=2)
        pd.DataFrame([xgb_metrics]).to_csv(
            out_dir / "xgboost_metrics.csv", index=False
        )

        # -----------------------------------------------------------------
        # Direct Q-TeleGuard vs XGBoost comparison.
        # -----------------------------------------------------------------
        comparison_rows = [
            ("Accuracy", "accuracy"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Macro Precision", "macro_precision"),
            ("Macro Recall", "macro_recall"),
            ("Macro F1-score", "macro_f1"),
            ("Macro PR-AUC", "macro_pr_auc"),
            ("Macro ROC-AUC", "macro_roc_auc_ovr"),
            ("Sensitivity", "sensitivity"),
            ("Overall Performance", "overall_performance"),
        ]

        comparison_data = []
        for display_name, key in comparison_rows:
            comparison_data.append({
                "Metric": display_name,
                "Q-TeleGuard": q_metrics.get(key, np.nan),
                "XGBoost": xgb_metrics.get(key, np.nan),
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(
            out_dir / "q_teleguard_vs_xgboost_metrics.csv", index=False
        )

        # Grouped performance comparison graph.
        plot_df = comparison_df.dropna().copy()
        if not plot_df.empty:
            x = np.arange(len(plot_df))
            width = 0.36

            plt.figure(figsize=(12, 6))
            plt.bar(
                x - width / 2,
                plot_df["Q-TeleGuard"],
                width,
                label="Q-TeleGuard",
            )
            plt.bar(
                x + width / 2,
                plot_df["XGBoost"],
                width,
                label="XGBoost",
            )
            plt.xticks(x, plot_df["Metric"], rotation=25, ha="right")
            plt.ylim(0, 1.05)
            plt.ylabel("Score")
            plt.title("Q-TeleGuard vs XGBoost - Performance Comparison")
            plt.legend()
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                out_dir / "14_q_teleguard_vs_xgboost_comparison.png",
                dpi=150,
            )
            plt.close()

        # Efficiency comparison: training time and prediction latency.
        efficiency_df = pd.DataFrame([
            {"Metric": "Training Time (seconds)", "Q-TeleGuard": q_metrics.get("training_time_seconds", np.nan), "XGBoost": xgb_metrics.get("training_time_seconds", np.nan)},
            {"Metric": "Prediction Latency (ms/record)", "Q-TeleGuard": q_metrics.get("prediction_latency_ms_per_record", np.nan), "XGBoost": xgb_metrics.get("prediction_latency_ms_per_record", np.nan)},
        ])
        efficiency_df.to_csv(out_dir / "q_teleguard_vs_xgboost_efficiency.csv", index=False)

        x_eff = np.arange(len(efficiency_df))
        width_eff = 0.36
        plt.figure(figsize=(9, 5))
        plt.bar(x_eff - width_eff / 2, efficiency_df["Q-TeleGuard"], width_eff, label="Q-TeleGuard")
        plt.bar(x_eff + width_eff / 2, efficiency_df["XGBoost"], width_eff, label="XGBoost")
        plt.xticks(x_eff, efficiency_df["Metric"], rotation=15, ha="right")
        plt.ylabel("Time")
        plt.title("Q-TeleGuard vs XGBoost - Training Time and Prediction Latency")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "14b_q_teleguard_vs_xgboost_efficiency.png", dpi=150)
        plt.close()

        # Confusion matrices side-by-side are intentionally not merged into one
        # figure; save the XGBoost confusion matrix separately for inspection.
        cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1, 2])
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix - XGBoost")
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(
            tick_marks,
            [ACTIONS[i] for i in range(3)],
            rotation=20,
        )
        plt.yticks(tick_marks, [ACTIONS[i] for i in range(3)])
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(3):
            for j in range(3):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )
        plt.ylabel("True class")
        plt.xlabel("Predicted class")
        plt.tight_layout()
        plt.savefig(out_dir / "15_confusion_matrix_xgboost.png", dpi=150)
        plt.close()

        # Feature-importance graph helps explain the benchmark.
        importances = pd.Series(
            model.feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)

        importances.to_csv(
            out_dir / "xgboost_feature_importance.csv",
            header=["importance"],
        )

        top_imp = importances.head(15).sort_values()
        plt.figure(figsize=(9, 7))
        plt.barh(top_imp.index, top_imp.values)
        plt.xlabel("Feature importance")
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        plt.savefig(out_dir / "16_xgboost_feature_importance.png", dpi=150)
        plt.close()

        return xgb_metrics

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: Path):
        payload = {
            "config": asdict(self.cfg),
            "q_table": dict(self.q_table),
            "bin_edges": self.bin_edges,
            "label_source": self.label_source,
            "pseudo_thresholds": getattr(self, "pseudo_thresholds", None),
            "training_history": self.training_history,
            "state_visit_counts": dict(self.state_visit_counts),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "QTeleGuard":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(QConfig(**payload["config"]))
        obj.q_table = defaultdict(
            lambda: np.zeros(len(ACTIONS), dtype=np.float64),
            payload["q_table"],
        )
        obj.bin_edges = payload["bin_edges"]
        obj.label_source = payload.get("label_source", "UNKNOWN")
        obj.pseudo_thresholds = payload.get("pseudo_thresholds")
        obj.training_history = payload.get("training_history", obj.training_history)

        # Backward-compatible loading: older saved models will not contain state
        # visit counts or the additional training-history keys.
        obj.state_visit_counts = defaultdict(
            int,
            payload.get("state_visit_counts", {}),
        )
        for key in (
            "new_states",
            "cumulative_visited_states",
            "state_coverage_percent",
        ):
            if key not in obj.training_history:
                obj.training_history[key] = []

        return obj


def chronological_split(df: pd.DataFrame, train_fraction: float):
    """
    Chronological split by global CDR time to reduce leakage from future behavior.
    """
    ordered = df.sort_values("CALL_START_DATE").reset_index(drop=True)
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def main():
    parser = argparse.ArgumentParser(description="Q-TeleGuard strictly Q-learning telecom fraud detector")
    parser.add_argument("--input", required=True, help="Path to exported Excel CDR file (.xlsx)")
    parser.add_argument("--sheet", default="0", help="Excel sheet name or zero-based index (default 0)")
    parser.add_argument("--output-dir", default="q_teleguard_output")
    parser.add_argument("--label-column", default=None, help="Optional ground-truth fraud label column")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.15)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = QConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    agent = QTeleGuard(cfg)
    raw = agent.load_excel(args.input, sheet_name=sheet)
    df = agent.engineer_features(raw)

    if len(df) < 10:
        raise ValueError("Too few valid CDR rows after preprocessing; at least 10 are recommended.")

    train_df, test_df = chronological_split(df, cfg.train_fraction)

    # Fit discretization and pseudo-label thresholds on TRAINING data only.
    agent.fit_state_bins(train_df)
    agent._fit_pseudo_thresholds(train_df)

    if args.label_column:
        label_col = args.label_column.strip().upper()
        if label_col not in df.columns:
            raise ValueError(
                f"--label-column {label_col!r} not found. Available columns: {list(df.columns)}"
            )
        df["TARGET"] = df[label_col].apply(agent._normalize_label)
        agent.label_source = f"GROUND_TRUTH:{label_col}"
        train_df, test_df = chronological_split(df, cfg.train_fraction)
    else:
        train_df["TARGET"] = agent.create_pseudo_labels(train_df)
        test_df["TARGET"] = agent.create_pseudo_labels(test_df)
        agent.label_source = "PSEUDO_LABELS_FROM_CDR_RULES"

    # Save label distribution so class imbalance is visible.
    label_summary = pd.concat(
        [
            train_df["TARGET"].value_counts().rename("train_count"),
            test_df["TARGET"].value_counts().rename("test_count"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    label_summary.index = [ACTIONS.get(int(i), str(i)) for i in label_summary.index]
    label_summary.to_csv(out_dir / "label_distribution.csv")

    print("\nQ-TeleGuard configuration")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"\nLabel source: {agent.label_source}")
    print("\nTraining label distribution:")
    print(train_df["TARGET"].map(ACTIONS).value_counts(dropna=False))
    print("\nTesting label distribution:")
    print(test_df["TARGET"].map(ACTIONS).value_counts(dropna=False))
    print()

    q_train_start = time.perf_counter()
    agent.train(train_df)
    q_training_time = time.perf_counter() - q_train_start

    metrics = agent.evaluate(test_df, out_dir)
    metrics["training_time_seconds"] = float(q_training_time)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    agent.save(out_dir / "q_teleguard_model.pkl")

    print("\nTraining XGBoost benchmark for comparison...")
    xgb_metrics = agent.train_xgboost_benchmark(
        train_df=train_df,
        test_df=test_df,
        q_metrics=metrics,
        out_dir=out_dir,
    )

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": str(args.input),
                "sheet": sheet,
                "rows_after_preprocessing": int(len(df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "label_source": agent.label_source,
                "config": asdict(cfg),
                "state_bins": agent.bin_edges,
                "pseudo_thresholds": getattr(agent, "pseudo_thresholds", None),
            },
            f,
            indent=2,
            default=str,
        )

    print("\nQ-TeleGuard evaluation metrics")
    print(json.dumps(metrics, indent=2))

    print("\nXGBoost benchmark metrics")
    print(json.dumps(xgb_metrics, indent=2))

    print(
        "\nComparison graph: "
        + str(out_dir / "14_q_teleguard_vs_xgboost_comparison.png")
    )
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
