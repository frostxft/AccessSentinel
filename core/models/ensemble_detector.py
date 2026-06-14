"""Ensemble anomaly detector combining IsolationForest, OneClassSVM, and
LocalOutlierFactor with majority voting.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "models")

DEFAULT_CONTAMINATION: float = 0.15
DEFAULT_SVM_NU: float = 0.05
DEFAULT_N_NEIGHBORS: int = 20
RANDOM_STATE: int = 42


class EnsembleAnomalyDetector:
    """Ensemble of three anomaly detectors using majority voting.

    Combines IsolationForest, OneClassSVM, and LocalOutlierFactor.  A sample
    is flagged as anomalous when at least two of the three detectors agree.
    Anomaly scores are computed by averaging per-detector scores that have
    been min-max normalized to the [0, 1] range.

    Args:
        name: Logical name used for persisting / loading model files.
        contamination: Expected proportion of outliers (used by IF and LOF).
        svm_nu: Upper bound on training errors for OneClassSVM.

    Attributes:
        detector_if: IsolationForest instance.
        detector_svm: OneClassSVM instance.
        detector_lof: LocalOutlierFactor instance (novelty mode).
        name: Model name used for pkl file naming.
        is_fitted: Whether the ensemble has been fitted.
    """

    def __init__(
        self,
        name: str = "default",
        contamination: float = DEFAULT_CONTAMINATION,
        svm_nu: float = DEFAULT_SVM_NU,
    ) -> None:
        self.name = name
        self.detector_if = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.detector_svm = OneClassSVM(
            nu=svm_nu,
            kernel="rbf",
            gamma="auto",
        )
        self.detector_lof = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=DEFAULT_N_NEIGHBORS,
            novelty=True,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _detector_path(self, detector_name: str) -> str:
        """Build the filesystem path for a persisted detector pkl file.

        Args:
            detector_name: Short label such as ``"if"``, ``"svm"``, ``"lof"``,
                or ``"scaler"``.

        Returns:
            Absolute path to the ``.pkl`` file.
        """
        return os.path.join(MODELS_DIR, f"{self.name}_{detector_name}.pkl")

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fit (or load) all three detectors and scaler on the supplied matrix.

        If pkl files for all three detectors and the scaler already exist under
        :data:`MODELS_DIR`, they are loaded from disk instead of being
        re-trained.  After a successful fit the detectors and scaler are
        persisted so that subsequent runs can skip training.

        Feature matrix is standardized (zero mean, unit variance) before
        fitting the scale-sensitive detectors (OneClassSVM, LocalOutlierFactor).

        Args:
            feature_matrix: ``(n_samples, n_features)`` array of type float64.
        """
        if_path = self._detector_path("if")
        svm_path = self._detector_path("svm")
        lof_path = self._detector_path("lof")
        scaler_path = self._detector_path("scaler")

        if all(os.path.exists(p) for p in (if_path, svm_path, lof_path, scaler_path)):
            self.detector_if = joblib.load(if_path)
            self.detector_svm = joblib.load(svm_path)
            self.detector_lof = joblib.load(lof_path)
            self.scaler = joblib.load(scaler_path)
        else:
            os.makedirs(MODELS_DIR, exist_ok=True)
            scaled = self.scaler.fit_transform(feature_matrix)
            self.detector_if.fit(scaled)
            self.detector_svm.fit(scaled)
            self.detector_lof.fit(scaled)
            joblib.dump(self.detector_if, if_path)
            joblib.dump(self.detector_svm, svm_path)
            joblib.dump(self.detector_lof, lof_path)
            joblib.dump(self.scaler, scaler_path)

        self.is_fitted = True

    def predict(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and compute aggregate anomaly scores.

        Each base detector returns ``+1`` (inlier / normal) or ``-1``
        (outlier / anomaly).  A sample is considered anomalous when at least
        two out of the three detectors vote anomaly.

        The anomaly score is the average of per-detector scores that have
        been inverted (so higher means more anomalous) and min-max
        normalized to ``[0, 1]`` across the batch.

        Args:
            feature_matrix: ``(n_samples, n_features)`` array of type float64.

        Returns:
            A tuple ``(is_anomaly, anomaly_score)`` where:

            * ``is_anomaly`` -- boolean array of shape ``(n_samples,)``.
            * ``anomaly_score`` -- float64 array of shape ``(n_samples,)``
              with values in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("EnsembleAnomalyDetector has not been fitted.")

        # Apply StandardScaler before prediction
        scaled = self.scaler.transform(feature_matrix)

        raw_if = self.detector_if.decision_function(scaled)
        raw_svm = self.detector_svm.decision_function(scaled)
        raw_lof = self.detector_lof.decision_function(scaled)

        norm_if = self._to_anomaly_score(raw_if)
        norm_svm = self._to_anomaly_score(raw_svm)
        norm_lof = self._to_anomaly_score(raw_lof)

        anomaly_score = (norm_if + norm_svm + norm_lof) / 3.0

        pred_if = self.detector_if.predict(scaled)
        pred_svm = self.detector_svm.predict(scaled)
        pred_lof = self.detector_lof.predict(scaled)

        vote_sum = pred_if + pred_svm + pred_lof
        is_anomaly = vote_sum <= -1

        return is_anomaly, anomaly_score

    def _to_anomaly_score(self, detector_scores: np.ndarray) -> np.ndarray:
        """Convert raw detector scores to normalized anomaly scores.

        Raw scores from each detector follow the convention *higher = more
        normal / inlier*.  This helper negates them so that *higher = more
        anomalous*, then applies min-max normalization across the batch.

        Args:
            detector_scores: Raw ``decision_function`` output of shape
                ``(n_samples,)``.

        Returns:
            Float64 array of shape ``(n_samples,)`` with values in [0, 1].
        """
        anomaly_scores = -detector_scores
        return self._minmax_normalize(anomaly_scores)

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize an array to the [0, 1] range.

        Args:
            scores: 1-d array of shape ``(n_samples,)``.

        Returns:
            Normalized float64 array.  If all values are equal the result
            is an array of zeros.
        """
        s_min = scores.min()
        s_max = scores.max()
        if s_max - s_min < 1e-9:
            return np.zeros_like(scores, dtype=np.float64)
        return (scores - s_min) / (s_max - s_min)
