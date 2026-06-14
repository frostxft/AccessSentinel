"""Random Forest models for anomaly risk prediction and access decisions.

Provides two RandomForest-based models:
- :class:`RiskClassifier`: predicts anomaly risk (binary classification).
- :class:`AccessPredictor`: makes access control decisions (APPROVE / DENY / REVIEW).
"""

import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "models"
)
RANDOM_STATE: int = 42
MIN_AUC: float = 0.75


# ── Custom Exception ──────────────────────────────────────────────────────────


class ModelQualityError(Exception):
    """Raised when a trained model's AUC falls below the minimum threshold.

    Attributes:
        actual_auc: The AUC score that triggered the error.
    """

    def __init__(self, actual_auc: float) -> None:
        self.actual_auc = actual_auc
        super().__init__(
            f"Model AUC {actual_auc:.4f} is below minimum {MIN_AUC:.2f}"
        )


# ── RiskClassifier ───────────────────────────────────────────────────────────


class RiskClassifier:
    """Random Forest classifier for anomaly risk prediction.

    Trains a binary classifier on feature matrices with optional SMOTE
    oversampling for imbalanced datasets.  Caches the trained model to disk.

    Attributes:
        model: The underlying sklearn RandomForestClassifier.
        is_fitted: Whether the model has been trained or loaded from cache.
        feature_importances: Top-3 (feature_name, importance) pairs.
    """

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_importances: list[tuple[str, float]] = []

    def fit(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the model on the given feature matrix and labels.

        Loads a cached model from ``models/random_forest.pkl`` if available.
        Otherwise applies SMOTE oversampling when the positive-class ratio is
        below 0.20, performs an 80/20 train/test split, trains, validates
        AUC, stores top-3 feature importances, and persists the model.

        Args:
            feature_matrix: 2-D array of shape (n_samples, n_features).
            labels: 1-D array of binary labels (0 = normal, 1 = anomaly).
            feature_names: Optional names for each feature column.  Used to
                label the stored feature importances.

        Raises:
            ModelQualityError: If the test-set AUC falls below ``MIN_AUC``.
        """
        model_path = os.path.join(MODELS_DIR, "random_forest.pkl")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_fitted = True
            importances = self.model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                named = list(zip(feature_names, importances))
                named.sort(key=lambda x: x[1], reverse=True)
                self.feature_importances = named[:3]
            return

        pos_ratio = float(np.sum(labels)) / float(len(labels))
        if pos_ratio < 0.20:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=RANDOM_STATE)
            feature_matrix, labels = smote.fit_resample(feature_matrix, labels)

        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            labels,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        if auc < MIN_AUC:
            raise ModelQualityError(auc)

        importances = self.model.feature_importances_
        if feature_names and len(feature_names) == len(importances):
            named = list(zip(feature_names, importances))
            named.sort(key=lambda x: x[1], reverse=True)
            self.feature_importances = named[:3]
        else:
            indices = np.argsort(importances)[::-1][:3]
            self.feature_importances = [
                (str(i), float(importances[i])) for i in indices
            ]

        self.is_fitted = True
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, model_path)

    def predict_proba(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return the probability of anomaly (class 1) for each sample.

        Args:
            feature_matrix: 2-D array of shape (n_samples, n_features).

        Returns:
            1-D array of probabilities for the positive (anomaly) class.
        """
        return self.model.predict_proba(feature_matrix)[:, 1]

    def get_top_features(self) -> list[tuple[str, float]]:
        """Return the top-3 most important features with their importance scores.

        Returns:
            List of (feature_name, importance) tuples sorted descending by
            importance.  Returns an empty list if the model has not been fitted
            or no feature names were supplied during training.
        """
        return self.feature_importances


# ── AccessDecision ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccessDecision:
    """The outcome of an access control prediction.

    Attributes:
        decision: One of ``APPROVE``, ``DENY``, or ``REVIEW``.
        confidence: Model confidence for the predicted decision (0.0 – 1.0).
        peer_comparison_note: Human-readable note comparing the request to
            peer-group access patterns.
    """

    decision: str
    confidence: float
    peer_comparison_note: str


# ── AccessPredictor ──────────────────────────────────────────────────────────


class AccessPredictor:
    """Random Forest classifier for access control decisions.

    Trains a multi-class classifier (APPROVE=0, DENY=1, REVIEW=2) on
    one-hot encoded categorical features and a numerical peer-access-rate
    feature.  Caches the trained model and encoding metadata to disk.

    Attributes:
        model: The underlying sklearn RandomForestClassifier.
        is_fitted: Whether the model has been trained or loaded from cache.
        feature_columns_: Column names of the one-hot encoded training
            DataFrame, stored after fitting for prediction alignment.
    """

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_columns_: list[str] = []

    def fit(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Train the access predictor on historical access records.

        Loads a cached model from ``models/access_predictor.pkl`` if
        available.  Otherwise one-hot encodes the categorical columns
        (department, job_title, resource, action), retains the
        peer_access_rate numeric column, trains the classifier, and
        persists the model together with encoding metadata.

        Labels must be encoded as: APPROVE=0, DENY=1, REVIEW=2.

        Args:
            features_df: DataFrame with columns ``department``,
                ``job_title``, ``resource``, ``action`` (categorical),
                and ``peer_access_rate`` (float).
            labels: Series of integer-encoded decision labels (0, 1, or 2).
        """
        model_path = os.path.join(MODELS_DIR, "access_predictor.pkl")

        if os.path.exists(model_path):
            cached = joblib.load(model_path)
            self.model = cached["model"]
            self.feature_columns_ = cached["feature_columns"]
            self.is_fitted = True
            return

        categorical_cols = ["department", "job_title", "resource", "action"]
        X_encoded = pd.get_dummies(features_df, columns=categorical_cols)
        self.feature_columns_ = list(X_encoded.columns)

        self.model.fit(X_encoded, labels)
        self.is_fitted = True

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns_,
            },
            model_path,
        )

    def predict(
        self,
        department: str,
        job_title: str,
        resource: str,
        action: str,
        peer_access_rate: float,
    ) -> AccessDecision:
        """Predict an access control decision for a single request.

        If the model has not been fitted, a default ``REVIEW`` decision is
        returned.  Unknown categorical values (not seen during training) are
        silently encoded as all-zeros for those dummied columns.

        Args:
            department: The requestor's department.
            job_title: The requestor's job title.
            resource: The resource being accessed.
            action: The action being performed.
            peer_access_rate: Fraction of department peers who have accessed
                this resource with this action.

        Returns:
            An :class:`AccessDecision` with the predicted outcome and
            confidence.
        """
        if not self.is_fitted:
            return AccessDecision(
                decision="REVIEW",
                confidence=0.5,
                peer_comparison_note=(
                    "Model not fitted; defaulting to REVIEW."
                ),
            )

        input_df = pd.DataFrame(
            [
                {
                    "department": department,
                    "job_title": job_title,
                    "resource": resource,
                    "action": action,
                    "peer_access_rate": peer_access_rate,
                }
            ]
        )

        input_encoded = pd.get_dummies(
            input_df,
            columns=["department", "job_title", "resource", "action"],
        )

        for col in self.feature_columns_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[self.feature_columns_]

        proba = self.model.predict_proba(input_encoded)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        decision_map: dict[int, str] = {0: "APPROVE", 1: "DENY", 2: "REVIEW"}
        decision = decision_map.get(pred_class, "REVIEW")

        note = (
            f"Peer access rate for {resource}/{action} in {department} "
            f"is {peer_access_rate:.2%}."
        )

        return AccessDecision(
            decision=decision,
            confidence=confidence,
            peer_comparison_note=note,
        )


# ── Peer Access Rate Helper ──────────────────────────────────────────────────


def compute_peer_access_rate(
    resource: str,
    action: str,
    department: str,
    events_df: pd.DataFrame,
) -> float:
    """Compute the fraction of department peers who accessed a resource/action.

    The peer access rate is defined as the proportion of users in the given
    department who have performed the specified action on the specified
    resource.

    Args:
        resource: The resource name to filter on.
        action: The action type to filter on.
        department: The department whose users constitute the peer group.
        events_df: DataFrame with columns ``user_id``, ``department``,
            ``resource``, and ``action``.

    Returns:
        A float between 0.0 and 1.0 representing the peer access rate.
        Returns 0.0 if the events DataFrame is empty or the department has
        no users.
    """
    if events_df.empty:
        return 0.0

    required_cols = {"user_id", "department", "resource", "action"}
    if not required_cols.issubset(events_df.columns):
        return 0.0

    dept_mask = events_df["department"] == department
    if not dept_mask.any():
        return 0.0

    dept_users = set(events_df.loc[dept_mask, "user_id"].unique())
    if not dept_users:
        return 0.0

    resource_action_mask = (
        (events_df["resource"] == resource)
        & (events_df["action"] == action)
    )
    users_with_access = set(
        events_df.loc[resource_action_mask, "user_id"].unique()
    )

    return float(len(users_with_access & dept_users) / len(dept_users))
