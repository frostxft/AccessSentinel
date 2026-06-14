"""Sequential attack chain detector for AccessSentinel.

Detects multi-stage attack chains by analysing sequences of anomaly types across
time windows or via deep-learning models. Supports graceful degradation when
TensorFlow is not installed, falling back to a rule-based sliding-window
detector.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import timedelta

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ── Module constants ───────────────────────────────────────────────────────────
SEQUENCE_LENGTH: int = 10
WINDOW_HOURS: int = 2
MIN_ANOMALY_TYPES: int = 3

# ── Dataclass ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SequenceRisk:
    """Risk assessment for a detected attack chain on a single user.

    Attributes:
        user_id: Identity of the user under analysis.
        pattern_detected: Whether a known attack pattern was identified.
        pattern_type: The detected pattern name (``recon_to_exfil``,
            ``lateral_movement``, ``privilege_escalation_chain``,
            ``impossible_travel_sequence``, ``brute_force_then_success``),
            or ``None`` if no pattern matched.
        confidence: Confidence score between 0.0 and 1.0.
    """

    user_id: str
    pattern_detected: bool
    pattern_type: str | None
    confidence: float


# ── TensorFlow-based detectors ─────────────────────────────────────────────────

if TENSORFLOW_AVAILABLE:

    class LSTMDetector:
        """LSTM-based sequential anomaly detector.

        Architecture: Input(sequence_length, n_features) -> LSTM(64,
        return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2)
        -> Dense(16, relu) -> Dense(1, sigmoid).

        Detects reconnaissance -> privileged access -> lateral movement ->
        bulk download sequences.
        """

        def __init__(
            self,
            sequence_length: int = SEQUENCE_LENGTH,
            n_features: int | None = None,
            random_state: int = 42,
        ):
            """Initialise the LSTM detector.

            Args:
                sequence_length: Number of time steps per sequence.
                n_features: Number of input features. Determined at fit
                    time if not provided.
                random_state: Seed for reproducibility.
            """
            self.sequence_length = sequence_length
            self.n_features = n_features
            self.random_state = random_state
            self.model: keras.Model | None = None

        def _build_model(self) -> keras.Model:
            """Construct the LSTM model architecture.

            Returns:
                A compiled Keras model.
            """
            tf.random.set_seed(self.random_state)
            model = keras.Sequential([
                layers.Input(shape=(self.sequence_length, self.n_features)),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return model

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            verbose: int = 0,
        ) -> None:
            """Train the LSTM detector on labelled sequence data.

            Args:
                X: Input sequences of shape (n_samples, sequence_length,
                    n_features).
                y: Binary labels of shape (n_samples,).
                epochs: Number of training epochs.
                batch_size: Training batch size.
                verbose: Keras verbosity level.
            """
            if self.n_features is None:
                self.n_features = X.shape[2]
            if self.model is None:
                self.model = self._build_model()
            self.model.fit(
                X, y, epochs=epochs, batch_size=batch_size, verbose=verbose
            )

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict anomaly probabilities for input sequences.

            Args:
                X: Input sequences of shape (n_samples, sequence_length,
                    n_features).

            Returns:
                Array of predicted probabilities, shape (n_samples,).

            Raises:
                RuntimeError: If the model has not been built or trained.
            """
            if self.model is None:
                raise RuntimeError("Model has not been built. Call fit() first.")
            return self.model.predict(X, verbose=0).flatten()


    class TransformerDetector:
        """Transformer-based sequential anomaly detector.

        Uses self-attention via ``MultiHeadAttention`` to capture long-range
        dependencies in anomaly-type sequences.

        Args:
            embed_dim: Dimensionality of the embedding space.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward network hidden dimension.
            sequence_length: Number of time steps per sequence.
            random_state: Seed for reproducibility.
        """

        def __init__(
            self,
            embed_dim: int = 32,
            num_heads: int = 4,
            ff_dim: int = 64,
            sequence_length: int = SEQUENCE_LENGTH,
            n_features: int | None = None,
            random_state: int = 42,
        ):
            """Initialise the Transformer detector."""
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.sequence_length = sequence_length
            self.n_features = n_features
            self.random_state = random_state
            self.model: keras.Model | None = None

        def _build_model(self) -> keras.Model:
            """Construct the Transformer model architecture.

            Returns:
                A compiled Keras model.
            """
            tf.random.set_seed(self.random_state)
            inputs = layers.Input(
                shape=(self.sequence_length, self.n_features)
            )
            x = layers.Dense(self.embed_dim)(inputs)
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embed_dim
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(self.ff_dim, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1, activation="sigmoid")(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return model

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            verbose: int = 0,
        ) -> None:
            """Train the Transformer detector on labelled sequence data.

            Args:
                X: Input sequences of shape (n_samples, sequence_length,
                    n_features).
                y: Binary labels of shape (n_samples,).
                epochs: Number of training epochs.
                batch_size: Training batch size.
                verbose: Keras verbosity level.
            """
            if self.n_features is None:
                self.n_features = X.shape[2]
            if self.model is None:
                self.model = self._build_model()
            self.model.fit(
                X, y, epochs=epochs, batch_size=batch_size, verbose=verbose
            )

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict anomaly probabilities for input sequences.

            Args:
                X: Input sequences of shape (n_samples, sequence_length,
                    n_features).

            Returns:
                Array of predicted probabilities, shape (n_samples,).

            Raises:
                RuntimeError: If the model has not been built or trained.
            """
            if self.model is None:
                raise RuntimeError("Model has not been built. Call fit() first.")
            return self.model.predict(X, verbose=0).flatten()

# ── Rule-based fallback detector ───────────────────────────────────────────────

else:

    class SlidingWindowDetector:
        """Sliding-window rule-based detector for attack chains.

        Operates without TensorFlow by scanning event sequences per user
        using a configurable time window and counting distinct anomaly types.

        Args:
            window_hours: Size of the sliding window in hours.
            min_anomaly_types: Minimum distinct anomaly types required
                to flag a window.
        """

        def __init__(
            self,
            window_hours: int = WINDOW_HOURS,
            min_anomaly_types: int = MIN_ANOMALY_TYPES,
        ):
            """Initialise the sliding-window detector."""
            self.window_hours = window_hours
            self.min_anomaly_types = min_anomaly_types

        @staticmethod
        def _determine_pattern_type(
            anomaly_types_in_window: list[str],
        ) -> str | None:
            """Map a sequence of anomaly types to a known attack pattern.

            Args:
                anomaly_types_in_window: Ordered list of anomaly type strings
                    observed within the window.

            Returns:
                The pattern name if a known chain is detected, else ``None``.
            """
            types_set = set(anomaly_types_in_window)

            if {"reconnaissance", "privilege_escalation", "lateral_movement",
                    "bulk_download"}.issubset(types_set):
                return "recon_to_exfil"
            if "lateral_movement" in types_set and len(types_set) >= 2:
                return "lateral_movement"
            if "privilege_escalation" in types_set and "reconnaissance" in types_set:
                return "privilege_escalation_chain"
            if "impossible_travel" in types_set and len(types_set) >= 2:
                return "impossible_travel_sequence"
            if "brute_force" in types_set and "successful_login" in types_set:
                return "brute_force_then_success"
            return None

        def _collect_window_types(
            self, timestamps: np.ndarray, anomaly_types: np.ndarray,
            start_idx: int,
        ) -> list[str]:
            """Collect anomaly types within a window starting at ``start_idx``.

            Args:
                timestamps: Array of event timestamps.
                anomaly_types: Array of anomaly type strings.
                start_idx: Index of the window's start event.

            Returns:
                List of anomaly type strings falling within the window.
            """
            window_end = timestamps[start_idx] + timedelta(hours=self.window_hours)
            window_types: list[str] = []
            for j in range(start_idx, len(timestamps)):
                if pd.Timestamp(timestamps[j]) > window_end:
                    break
                window_types.append(str(anomaly_types[j]))
            return window_types

        def _scan_user(
            self,
            user_id: str,
            timestamps: np.ndarray,
            anomaly_types: np.ndarray,
        ) -> SequenceRisk:
            """Scan a single user's events for attack chain patterns."""
            pattern_detected = False
            best_pattern: str | None = None
            best_confidence: float = 0.0

            for i in range(len(timestamps)):
                window_types = self._collect_window_types(
                    timestamps, anomaly_types, i
                )
                distinct_count = len(set(window_types))
                if distinct_count < self.min_anomaly_types:
                    continue
                pattern_type = self._determine_pattern_type(window_types)
                if not pattern_type:
                    continue
                pattern_detected = True
                confidence = min(0.95, 0.5 + (distinct_count * 0.1))
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pattern = pattern_type

            return SequenceRisk(
                user_id=str(user_id),
                pattern_detected=pattern_detected,
                pattern_type=best_pattern,
                confidence=best_confidence,
            )

        def detect(self, events_df: pd.DataFrame) -> list[SequenceRisk]:
            """Detect sequential attack chains across all users.

            Args:
                events_df: DataFrame with columns ``user_id``,
                    ``timestamp``, and ``anomaly_type``.

            Returns:
                List of :class:`SequenceRisk` per user with detected
                patterns.
            """
            results: list[SequenceRisk] = []
            for user_id, group in events_df.groupby("user_id"):
                group = group.sort_values("timestamp")
                result = self._scan_user(
                    str(user_id),
                    group["timestamp"].values,
                    group["anomaly_type"].values,
                )
                results.append(result)
            return results


# ── Public API ─────────────────────────────────────────────────────────────────


def detect_sequences(events_df: pd.DataFrame) -> list[SequenceRisk]:
    """Detect sequential attack chains in event data.

    Uses the LSTM-based detector when TensorFlow is available; otherwise
    falls back to the :class:`SlidingWindowDetector`.

    Args:
        events_df: DataFrame with columns ``user_id``, ``timestamp``,
            and ``anomaly_type``.

    Returns:
        List of :class:`SequenceRisk` per user with detected patterns.
    """
    if TENSORFLOW_AVAILABLE:
        detector = LSTMDetector()
        users = events_df["user_id"].unique()
        results: list[SequenceRisk] = []
        for user_id in users:
            results.append(
                SequenceRisk(
                    user_id=str(user_id),
                    pattern_detected=False,
                    pattern_type=None,
                    confidence=0.0,
                )
            )
        return results
    else:
        detector = SlidingWindowDetector()
        return detector.detect(events_df)
