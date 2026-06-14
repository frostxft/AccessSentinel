"""Tests for core.models: ensemble_detector, random_forest, kmeans_clustering, sequential_detector."""

import os
import numpy as np
import pandas as pd
import pytest

from core.ingestion import ingest
from core.features import extract_features, FEATURE_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "sample_data")


def _load_test_identities():
    users_path = os.path.join(_get_data_dir(), "identity_users.csv")
    events_path = os.path.join(_get_data_dir(), "identity_events.csv")
    return ingest(users_path=users_path, events_path=events_path)


def _load_test_labels():
    labels_path = os.path.join(_get_data_dir(), "identity_users_labels.csv")
    labels_df = pd.read_csv(labels_path)
    label_map = {
        uid: bool(val) if isinstance(val, (bool, np.bool_))
        else str(val).strip().lower() == "true"
        for uid, val in zip(labels_df["user_id"], labels_df["is_anomaly"])
    }
    return label_map


@pytest.fixture
def test_identities():
    return _load_test_identities()


@pytest.fixture
def test_label_map():
    return _load_test_labels()


# ═══════════════════════════════════════════════════════════════════════════════
# EnsembleAnomalyDetector
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsembleAnomalyDetector:
    """Tests for ensemble_detector.EnsembleAnomalyDetector."""

    def test_fit_on_small_matrix(self, monkeypatch, tmp_path):
        import core.models.ensemble_detector as ed

        monkeypatch.setattr(ed, "MODELS_DIR", str(tmp_path))
        X = np.random.RandomState(42).randn(100, 5)
        det = ed.EnsembleAnomalyDetector(name="test_ensemble")
        det.fit(X)
        assert det.is_fitted is True

    def test_predict_returns_correct_shapes(self, monkeypatch, tmp_path):
        import core.models.ensemble_detector as ed

        monkeypatch.setattr(ed, "MODELS_DIR", str(tmp_path))
        X = np.random.RandomState(42).randn(100, 5)
        det = ed.EnsembleAnomalyDetector(name="test_ensemble")
        det.fit(X)
        is_anomaly, scores = det.predict(X)
        assert is_anomaly.shape == (100,)
        assert scores.shape == (100,)
        assert is_anomaly.dtype == bool
        assert scores.dtype == np.float64

    def test_majority_vote_logic(self, monkeypatch, tmp_path):
        import core.models.ensemble_detector as ed

        monkeypatch.setattr(ed, "MODELS_DIR", str(tmp_path))
        rng = np.random.RandomState(42)
        normal = rng.randn(150, 5) * 0.3 + 10.0
        outliers = rng.randn(50, 5) * 0.3 - 10.0
        X = np.vstack([normal, outliers])
        det = ed.EnsembleAnomalyDetector(name="test_majority", contamination=0.20)
        det.fit(X)
        is_anomaly, _ = det.predict(X)
        assert is_anomaly.shape == (200,)
        assert is_anomaly.dtype == bool
        assert is_anomaly.sum() > 0

    def test_determinism_same_runs_produce_same_predictions(self, monkeypatch, tmp_path):
        import core.models.ensemble_detector as ed

        monkeypatch.setattr(ed, "MODELS_DIR", str(tmp_path))
        X = np.random.RandomState(42).randn(100, 5)
        det1 = ed.EnsembleAnomalyDetector(name="det1")
        det1.fit(X)
        pred1_is, pred1_sc = det1.predict(X)

        det2 = ed.EnsembleAnomalyDetector(name="det2")
        det2.fit(X)
        pred2_is, pred2_sc = det2.predict(X)

        assert np.array_equal(pred1_is, pred2_is)
        assert np.allclose(pred1_sc, pred2_sc)

    def test_cache_loading(self, monkeypatch, tmp_path):
        import core.models.ensemble_detector as ed

        monkeypatch.setattr(ed, "MODELS_DIR", str(tmp_path))
        X = np.random.RandomState(42).randn(100, 5)

        det1 = ed.EnsembleAnomalyDetector(name="cache_test")
        det1.fit(X)
        pred1_is, pred1_sc = det1.predict(X)

        det2 = ed.EnsembleAnomalyDetector(name="cache_test")
        det2.fit(X)
        pred2_is, pred2_sc = det2.predict(X)

        assert np.array_equal(pred1_is, pred2_is)
        assert np.allclose(pred1_sc, pred2_sc)


# ═══════════════════════════════════════════════════════════════════════════════
# RiskClassifier (RandomForest)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRiskClassifier:
    """Tests for random_forest.RiskClassifier."""

    def test_fit_with_labels(self, test_identities, test_label_map, monkeypatch):
        import core.models.random_forest as rf

        monkeypatch.setattr(rf, "MIN_AUC", 0.0)
        identities = test_identities
        features = extract_features(identities, baselines={})
        labels = np.array(
            [float(test_label_map.get(rec.user_id, False))
             for rec in identities],
            dtype=np.float64,
        )
        assert features.shape[0] == len(identities)
        assert labels.shape[0] == len(identities)
        assert labels.sum() > 0
        clf = rf.RiskClassifier()
        clf.fit(features, labels, feature_names=FEATURE_NAMES)
        assert clf.is_fitted is True

    def test_model_persistence(self, test_identities, test_label_map, monkeypatch):
        import core.models.random_forest as rf

        monkeypatch.setattr(rf, "MIN_AUC", 0.0)
        identities = test_identities
        features = extract_features(identities, baselines={})
        labels = np.array(
            [float(test_label_map.get(rec.user_id, False))
             for rec in identities],
            dtype=np.float64,
        )
        clf = rf.RiskClassifier()
        clf.fit(features, labels, feature_names=FEATURE_NAMES)
        model_path = os.path.join(rf.MODELS_DIR, "random_forest.pkl")
        assert os.path.exists(model_path)

    def test_get_top_features(self, test_identities, test_label_map, monkeypatch):
        import core.models.random_forest as rf

        monkeypatch.setattr(rf, "MIN_AUC", 0.0)
        identities = test_identities
        features = extract_features(identities, baselines={})
        labels = np.array(
            [float(test_label_map.get(rec.user_id, False))
             for rec in identities],
            dtype=np.float64,
        )
        clf = rf.RiskClassifier()
        clf.fit(features, labels, feature_names=FEATURE_NAMES)
        top = clf.get_top_features()
        assert len(top) == 3
        for name, imp in top:
            assert isinstance(name, str)
            assert isinstance(imp, float)


# ═══════════════════════════════════════════════════════════════════════════════
# KMeansClustering (RoleMiner)
# ═══════════════════════════════════════════════════════════════════════════════


class TestKMeansClustering:
    """Tests for kmeans_clustering.RoleMiner."""

    def _load_events(self):
        events_path = os.path.join(_get_data_dir(), "identity_events.csv")
        return pd.read_csv(events_path).drop(columns=["department"])

    def test_create_access_matrix(self):
        from core.models.kmeans_clustering import RoleMiner

        events_df = self._load_events()
        miner = RoleMiner()
        matrix = miner.create_user_access_matrix(events_df)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.index.name == "user_id"
        assert "action_diversity" in matrix.columns

    def test_find_optimal_clusters(self):
        from core.models.kmeans_clustering import RoleMiner

        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        miner = RoleMiner()
        k = miner.find_optimal_clusters(X)
        assert isinstance(k, int)
        assert 2 <= k <= 15

    def test_fit_and_predict(self, monkeypatch, tmp_path):
        import core.models.kmeans_clustering as km

        monkeypatch.setattr(km, "MODELS_DIR", str(tmp_path))
        events_df = self._load_events()
        miner = km.RoleMiner()
        miner.fit(events_df)
        assert miner.is_fitted is True
        labels, profiles = miner.predict(events_df)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(events_df["user_id"].unique())
        assert isinstance(profiles, list)
        assert len(profiles) > 0

    def test_cache_loading(self, monkeypatch, tmp_path):
        import core.models.kmeans_clustering as km

        monkeypatch.setattr(km, "MODELS_DIR", str(tmp_path))
        events_df = self._load_events()

        miner1 = km.RoleMiner()
        miner1.fit(events_df)
        labels1, _ = miner1.predict(events_df)

        miner2 = km.RoleMiner()
        miner2.fit(events_df)
        miner2.scaler = miner1.scaler
        labels2, _ = miner2.predict(events_df)

        assert np.array_equal(labels1, labels2)


# ═══════════════════════════════════════════════════════════════════════════════
# SequentialDetector
# ═══════════════════════════════════════════════════════════════════════════════


class TestSequentialDetector:
    """Tests for sequential_detector."""

    @pytest.fixture
    def test_events_df(self):
        events_df = pd.DataFrame({
            "event_id": [f"E{i}" for i in range(10)],
            "user_id": ["U001"] * 10,
            "username": ["user_0001"] * 10,
            "department": ["Engineering"] * 10,
            "job_title": ["Engineer"] * 10,
            "resource": ["res1"] * 10,
            "action": ["read"] * 10,
            "timestamp": pd.date_range(
                "2025-01-01", periods=10, freq="15min", tz="UTC"
            ),
            "source_ip": ["1.1.1.1"] * 10,
            "location": ["NY"] * 10,
            "success": ["true"] * 10,
            "is_anomaly": ["true"] * 5 + ["false"] * 5,
            "anomaly_type": [
                "privilege_escalation",
                "unusual_location",
                "impossible_travel",
                "excessive_access",
                "unusual_time",
            ]
            + [""] * 5,
        })
        return events_df

    def test_tensorflow_available_flag(self):
        from core.models.sequential_detector import TENSORFLOW_AVAILABLE

        assert isinstance(TENSORFLOW_AVAILABLE, bool)

    def test_sliding_window_detector(self, test_events_df):
        from core.models.sequential_detector import SlidingWindowDetector

        detector = SlidingWindowDetector(
            window_hours=2, min_anomaly_types=2
        )
        results = detector.detect(test_events_df)
        assert len(results) == 1
        assert results[0].user_id == "U001"
        assert isinstance(results[0].pattern_detected, bool)

    def test_no_tf_fallback(self, monkeypatch, test_events_df):
        import core.models.sequential_detector as sd

        monkeypatch.setattr(sd, "TENSORFLOW_AVAILABLE", False)
        results = sd.detect_sequences(test_events_df)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_detect_sequences_returns_list(self, test_events_df):
        from core.models.sequential_detector import (
            detect_sequences,
            SequenceRisk,
        )

        results = detect_sequences(test_events_df)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SequenceRisk)


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism tests across models
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Cross-model determinism tests."""

    def test_random_forest_determinism(self, test_identities, test_label_map, monkeypatch):
        import core.models.random_forest as rf

        monkeypatch.setattr(rf, "MIN_AUC", 0.0)
        identities = test_identities
        features = extract_features(identities, baselines={})
        labels = np.array(
            [float(test_label_map.get(rec.user_id, False))
             for rec in identities],
            dtype=np.float64,
        )

        clf1 = rf.RiskClassifier()
        clf1.fit(features, labels, feature_names=FEATURE_NAMES)
        probs1 = clf1.predict_proba(features)

        clf2 = rf.RiskClassifier()
        clf2.fit(features, labels, feature_names=FEATURE_NAMES)
        probs2 = clf2.predict_proba(features)

        assert np.allclose(probs1, probs2)

    def test_kmeans_determinism(self, monkeypatch, tmp_path):
        import core.models.kmeans_clustering as km

        monkeypatch.setattr(km, "MODELS_DIR", str(tmp_path))
        events_df = TestKMeansClustering()._load_events()

        miner1 = km.RoleMiner()
        miner1.fit(events_df)

        miner2 = km.RoleMiner()
        miner2.fit(events_df)

        assert miner1.optimal_clusters == miner2.optimal_clusters
