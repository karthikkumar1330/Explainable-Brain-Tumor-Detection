from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from classification.domain.interfaces import IMetricsCalculator


class PyTorchMetricsCalculator(IMetricsCalculator):
    """Calculates classification metrics for multi-class tumor classification."""

    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, and F1 score (macro and weighted).

        Args:
            y_true: Array of ground-truth class labels.
            y_pred: Array of predicted class labels.

        Returns:
            A dictionary containing key performance metrics.
        """
        # Ensure input arrays are flat
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if len(y_true) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }

        accuracy = accuracy_score(y_true, y_pred)

        # Macro average
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Weighted average
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            # Default mapping for interface simplicity
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro),
        }
