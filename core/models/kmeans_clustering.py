"""K-Means clustering for role discovery and outlier detection.

Provides unsupervised role mining via K-Means on user access patterns,
automatic silhouette-based cluster optimization, and distance-based
outlier flagging.
"""

import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Module-level constants ────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MAX_CLUSTERS = 15
MIN_SILHOUETTE = 0.35
DEFAULT_CLUSTERS = 8
RANDOM_STATE = 42

# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class ClusterProfile:
    """Profile describing a discovered role cluster.

    Attributes:
        cluster_id: Integer identifier for the cluster.
        label: Human-readable label for the cluster.
        user_count: Number of users assigned to this cluster.
        avg_risk_score: Average risk score of users in this cluster.
        dominant_resources: Resources most frequently accessed by this
            cluster.
        dominant_actions: Actions most commonly performed by this cluster.
        outlier_count: Number of outliers detected in this cluster.
    """

    cluster_id: int
    label: str
    user_count: int
    avg_risk_score: float
    dominant_resources: list[str] = field(default_factory=list)
    dominant_actions: list[str] = field(default_factory=list)
    outlier_count: int = 0


# ── Helper ────────────────────────────────────────────────────────────────────


def _label_cluster(
    cluster_resources: pd.DataFrame,
    cluster_actions: pd.Series,
) -> str:
    """Generate a human-readable label for a cluster.

    Derives a label from the most dominant resource and action within the
    cluster.

    Args:
        cluster_resources: DataFrame of resource columns for cluster
            members.
        cluster_actions: Series of action frequencies for cluster members.

    Returns:
        A label string combining the top resource and action (e.g.
        ``"Finance_DB_read"``).
    """
    top_resource = (
        cluster_resources.mean().idxmax()
        if not cluster_resources.empty
        else "unknown"
    )
    top_action = (
        cluster_actions.idxmax()
        if not cluster_actions.empty
        else "unknown"
    )
    return f"{top_resource}_{top_action}"


# ── RoleMiner class ───────────────────────────────────────────────────────────


class RoleMiner:
    """K-Means clustering engine for role discovery and outlier detection.

    Discovers role clusters from user access patterns and flags users who
    deviate significantly from their cluster centroid as outliers.

    Attributes:
        kmeans: Fitted KMeans model instance.
        scaler: StandardScaler instance for feature normalization.
        optimal_clusters: Best number of clusters found during fitting.
        cluster_profiles: List of ClusterProfile instances from prediction.
        is_fitted: Whether the model has been successfully fitted.
        encoder_department: OneHotEncoder for department feature encoding.
        all_resources: List of all resource names seen during fitting.
        all_actions: List of all action names seen during fitting.
    """

    def __init__(self) -> None:
        """Initialize the RoleMiner with default configuration."""
        self.kmeans: KMeans | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.optimal_clusters: int = DEFAULT_CLUSTERS
        self.cluster_profiles: list[ClusterProfile] = []
        self.is_fitted = False
        self.encoder_department: OneHotEncoder | None = None
        self.all_resources: list[str] = []
        self.all_actions: list[str] = []

    # ── Public methods ────────────────────────────────────────────────────

    def create_user_access_matrix(
        self, events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build a normalized user access feature matrix.

        Pivots events by user and resource, adds action diversity and
        department encoding, then normalizes each row by total event count.

        Args:
            events_df: DataFrame with columns ``user_id``, ``resource``,
                ``action``, and optionally ``department``.

        Returns:
            A row-normalized DataFrame indexed by ``user_id``.
        """
        event_counts = (
            events_df.groupby(["user_id", "resource"])
            .size()
            .unstack(fill_value=0)
        )
        self.all_resources = sorted(event_counts.columns.tolist())

        action_diversity = (
            events_df.groupby("user_id")["action"]
            .nunique()
            .rename("action_diversity")
        )
        event_counts = event_counts.join(action_diversity)

        total_events = (
            events_df.groupby("user_id")
            .size()
            .rename("total_event_count")
        )
        event_counts = event_counts.join(total_events)

        if "department" in events_df.columns:
            event_counts = self._encode_department(events_df, event_counts)

        self.all_actions = sorted(events_df["action"].unique().tolist())

        totals = event_counts["total_event_count"]
        normalized = event_counts.drop(
            columns="total_event_count"
        ).div(totals, axis=0)

        return normalized

    def find_optimal_clusters(
        self, feature_matrix: np.ndarray
    ) -> int:
        """Determine the optimal number of clusters via silhouette analysis.

        Evaluates cluster counts from 2 to ``MAX_CLUSTERS`` and selects the
        configuration with the highest silhouette score above
        ``MIN_SILHOUETTE``, falling back to ``DEFAULT_CLUSTERS`` if none
        qualify.

        Args:
            feature_matrix: Scaled feature array of shape
                ``(n_users, n_features)``.

        Returns:
            The optimal number of clusters.
        """
        n_samples = feature_matrix.shape[0]
        max_k = min(MAX_CLUSTERS, n_samples - 1)
        if max_k < 2:
            return DEFAULT_CLUSTERS

        best_k = DEFAULT_CLUSTERS
        best_score = -1.0

        for k in range(2, max_k + 1):
            kmeans = KMeans(
                n_clusters=k, random_state=RANDOM_STATE, n_init=10
            )
            labels = kmeans.fit_predict(feature_matrix)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(feature_matrix, labels)
            if score > best_score:
                best_score = score
                best_k = k

        if best_score > MIN_SILHOUETTE:
            return best_k
        return DEFAULT_CLUSTERS

    def fit(self, events_df: pd.DataFrame) -> None:
        """Fit the K-Means model on user access events.

        Creates the user access feature matrix, standardizes features,
        determines the optimal cluster count, fits the KMeans model, and
        persists it to disk.

        Args:
            events_df: DataFrame with columns ``user_id``, ``resource``,
                ``action``, and optionally ``department``.
        """
        cache_path = os.path.join(MODELS_DIR, "kmeans.pkl")
        if os.path.exists(cache_path):
            self.kmeans = joblib.load(cache_path)
            self.is_fitted = True
            return

        matrix = self.create_user_access_matrix(events_df)
        features = self.scaler.fit_transform(matrix.values)
        self.optimal_clusters = self.find_optimal_clusters(features)
        self.kmeans = KMeans(
            n_clusters=self.optimal_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        self.kmeans.fit(features)

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.kmeans, cache_path)
        self.is_fitted = True

    def predict(
        self, events_df: pd.DataFrame
    ) -> tuple[np.ndarray, list[ClusterProfile]]:
        """Predict cluster assignments and build cluster profiles.

        Assigns users to clusters using the fitted KMeans model, labels
        each cluster by its dominant resource and action, flags outliers
        beyond two standard deviations from the centroid, and returns
        cluster profiles.

        Args:
            events_df: DataFrame with columns ``user_id``, ``resource``,
                ``action``, and optionally ``department`` and
                ``risk_score``.

        Returns:
            A tuple of ``(cluster_labels, cluster_profiles)`` where
            ``cluster_labels`` is an array of cluster assignments and
            ``cluster_profiles`` is a list of :class:`ClusterProfile`
            instances.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        matrix = self.create_user_access_matrix(events_df)
        features = self.scaler.transform(matrix.values)
        labels = self.kmeans.predict(features)

        user_risk = self._extract_risk_scores(events_df)
        profiles = self._build_profiles(
            features, labels, matrix, events_df, user_risk
        )
        self.cluster_profiles = profiles
        return labels, profiles

    # ── Private helpers ───────────────────────────────────────────────────

    def _encode_department(
        self, events_df: pd.DataFrame, event_counts: pd.DataFrame
    ) -> pd.DataFrame:
        """One-hot encode department column and join to event counts.

        Args:
            events_df: Original events DataFrame.
            event_counts: Pivoted user-resource matrix.

        Returns:
            DataFrame with department one-hot columns joined.
        """
        dept_series = events_df.groupby("user_id")["department"].first()
        if self.encoder_department is None:
            self.encoder_department = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            encoded = self.encoder_department.fit_transform(
                dept_series.values.reshape(-1, 1)
            )
        else:
            encoded = self.encoder_department.transform(
                dept_series.values.reshape(-1, 1)
            )
        dept_cols = self.encoder_department.get_feature_names_out(
            ["department"]
        )
        dept_df = pd.DataFrame(
            encoded, index=dept_series.index, columns=dept_cols
        )
        return event_counts.join(dept_df)

    def _extract_risk_scores(
        self, events_df: pd.DataFrame
    ) -> pd.Series:
        """Extract per-user risk scores from events DataFrame.

        Args:
            events_df: Events DataFrame optionally containing a
                ``risk_score`` column.

        Returns:
            Series of risk scores indexed by ``user_id``.
        """
        if "risk_score" in events_df.columns:
            return events_df.groupby("user_id")["risk_score"].mean()
        return pd.Series(
            data=0.0,
            index=events_df["user_id"].unique(),
            name="risk_score",
        )

    def _detect_outliers(
        self, cid: int, cluster_features: np.ndarray
    ) -> int:
        """Count outliers in a cluster based on centroid distance.

        Args:
            cid: Cluster identifier.
            cluster_features: Scaled features for cluster members.

        Returns:
            Number of users whose distance to centroid exceeds two
            standard deviations.
        """
        if len(cluster_features) == 0:
            return 0
        centroid = self.kmeans.cluster_centers_[cid]
        distances = np.linalg.norm(
            cluster_features - centroid, axis=1
        )
        threshold = distances.mean() + 2 * distances.std()
        return int((distances > threshold).sum())

    def _build_single_profile(
        self,
        cid: int,
        cluster_features: np.ndarray,
        cluster_matrix: pd.DataFrame,
        cluster_events: pd.DataFrame,
        user_risk: pd.Series,
        resource_cols: list[str],
    ) -> ClusterProfile:
        """Build a single ClusterProfile for one cluster.

        Args:
            cid: Cluster identifier.
            cluster_features: Scaled features for cluster members.
            cluster_matrix: Normalized access matrix for cluster members.
            cluster_events: Events for cluster members.
            user_risk: Per-user risk scores.
            resource_cols: Columns that represent resources.

        Returns:
            A populated ClusterProfile instance.
        """
        user_ids = cluster_matrix.index
        user_count = len(user_ids)

        outlier_count = self._detect_outliers(cid, cluster_features)

        resource_means = cluster_matrix[resource_cols].mean()
        top_resources = resource_means.nlargest(3).index.tolist()

        action_counts = cluster_events["action"].value_counts()
        top_actions = action_counts.nlargest(3).index.tolist()

        risk_values = user_risk.reindex(user_ids, fill_value=0.0)
        avg_risk = float(risk_values.mean())

        label = _label_cluster(
            cluster_matrix[resource_cols], action_counts
        )

        return ClusterProfile(
            cluster_id=int(cid),
            label=label,
            user_count=user_count,
            avg_risk_score=avg_risk,
            dominant_resources=top_resources,
            dominant_actions=top_actions,
            outlier_count=outlier_count,
        )

    def _build_profiles(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        matrix: pd.DataFrame,
        events_df: pd.DataFrame,
        user_risk: pd.Series,
    ) -> list[ClusterProfile]:
        """Build ClusterProfile instances for each discovered cluster.

        Args:
            features: Scaled feature array.
            labels: Cluster assignment for each user.
            matrix: Normalized user access matrix.
            events_df: Original events DataFrame.
            user_risk: Per-user risk scores.

        Returns:
            List of ClusterProfile instances, one per cluster.
        """
        resource_cols = [
            c for c in matrix.columns
            if c not in ("action_diversity",)
            and not c.startswith("department_")
        ]

        profiles = []
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            profiles.append(
                self._build_single_profile(
                    cid=int(cid),
                    cluster_features=features[mask],
                    cluster_matrix=matrix.loc[matrix.index[mask]],
                    cluster_events=events_df[
                        events_df["user_id"].isin(
                            matrix.index[mask]
                        )
                    ],
                    user_risk=user_risk,
                    resource_cols=resource_cols,
                )
            )
        return profiles
