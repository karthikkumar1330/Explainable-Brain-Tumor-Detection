import math
import numpy as np
from typing import List, Dict, Tuple
from research_framework.domain.entities import ModelPrediction, AgreementMetrics, EnsembleResult


class EnsembleEngine:
    """Computes soft-voting ensembles and calculates mathematical agreement indicators between models."""

    def compute_ensemble(self, predictions: List[ModelPrediction]) -> EnsembleResult:
        """Averages probability vectors across registered models to form an ensemble prediction.

        Args:
            predictions: List of model predictions.

        Returns:
            An EnsembleResult entity.
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty.")

        classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        avg_probs = {cls: 0.0 for cls in classes}

        for pred in predictions:
            for cls in classes:
                avg_probs[cls] += pred.probabilities.get(cls, 0.0)

        num_models = len(predictions)
        for cls in classes:
            avg_probs[cls] /= num_models

        # Select class with max averaged probability
        pred_class = max(avg_probs, key=avg_probs.get)
        ensemble_conf = avg_probs[pred_class]

        # Identify production prediction to compute agreement against
        production_pred = next((p for p in predictions if p.model_name == "efficientnet_b0"), None)
        agreement = None

        if production_pred and num_models > 1:
            agreement = self.calculate_agreement(production_pred, predictions)

        return EnsembleResult(
            predicted_class=pred_class,
            confidence=ensemble_conf,
            probabilities=avg_probs,
            individual_predictions=predictions,
            agreement_metrics=agreement
        )

    def calculate_agreement(self, production_pred: ModelPrediction, all_preds: List[ModelPrediction]) -> AgreementMetrics:
        """Computes Cosine Similarity and Jensen-Shannon Divergence compared to production baseline."""
        classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        eps = 1e-9

        # 1. Build production probability vector
        p_vec = np.array([production_pred.probabilities.get(cls, 0.0) for cls in classes])
        p_vec = p_vec / (p_vec.sum() + eps)

        # 2. Build research average probability vector
        research_preds = [p for p in all_preds if p.model_name != "efficientnet_b0"]
        if not research_preds:
            # Fallback if only production is loaded
            r_vec = p_vec
        else:
            r_vec = np.zeros(len(classes))
            for rp in research_preds:
                r_vec += np.array([rp.probabilities.get(cls, 0.0) for cls in classes])
            r_vec /= len(research_preds)
            r_vec = r_vec / (r_vec.sum() + eps)

        # 3. Calculate Cosine Similarity
        dot_product = np.dot(p_vec, r_vec)
        norm_p = np.linalg.norm(p_vec)
        norm_r = np.linalg.norm(r_vec)
        cosine_sim = float(dot_product / (norm_p * norm_r + eps))

        # 4. Calculate Jensen-Shannon Divergence
        # JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
        m_vec = 0.5 * (p_vec + r_vec)
        
        def kl_divergence(p, q):
            return sum(p[i] * math.log(p[i] / (q[i] + eps) + eps) for i in range(len(p)) if p[i] > 0)

        kl_pm = kl_divergence(p_vec, m_vec)
        kl_qm = kl_divergence(r_vec, m_vec)
        js_div = float(0.5 * kl_pm + 0.5 * kl_qm)

        # Classify agreement levels
        if cosine_sim >= 0.95:
            level = "HIGH CONSENSUS"
            message = "Multi-model research backbones show strong mathematical consensus with the production model."
        elif cosine_sim >= 0.82:
            level = "MODERATE CONSENSUS"
            message = "Mild variance detected in model logits. General diagnostic alignment exists."
        else:
            level = "DISCREPANCY WARNING"
            message = "Significant model divergence detected. Inspect quantitative overlays and differential statistics."

        return AgreementMetrics(
            cosine_similarity=cosine_sim,
            jensen_shannon_divergence=js_div,
            level=level,
            message=message
        )
