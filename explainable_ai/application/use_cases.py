import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import torch
from explainable_ai.domain.interfaces import IXAIEngine
from explainable_ai.domain.entities import ExplanationResult
from classification.domain.entities import BrainTumorClass


class GenerateExplanationUseCase:
    """Use case to generate Explainable AI 2.0 heatmaps, run spatial analytics,

    and produce natural language clinical explanations of model focus.
    """

    def __init__(self, xai_engine: IXAIEngine) -> None:
        """Initializes the explainability use case.

        Args:
            xai_engine: An instance of IXAIEngine.
        """
        self.xai_engine = xai_engine

    def execute(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        method: str = "gradcam",
        tumor_mask: Optional[np.ndarray] = None,
    ) -> ExplanationResult:
        """Generates explanation results with quadrant distributions and clinical text.

        Args:
            image_tensor: Input image tensor (C, H, W) or (1, C, H, W).
            target_class: Target class index to explain.
            method: The algorithm ('gradcam', 'gradcam_plus_plus', 'eigencam').
            tumor_mask: Optional binary tumor mask for overlap calculation.

        Returns:
            An ExplanationResult object.
        """
        # 1. Generate explanation heatmap using XAI engine
        heatmap = self.xai_engine.generate_explanation(
            image_tensor, target_class, method
        )

        # 2. Get user-friendly name of target class
        try:
            class_name = BrainTumorClass.get_name_by_value(target_class)
        except Exception:
            class_name = f"Class {target_class}"

        # 3. Analyze Quadrant Attention Distribution
        h, w = heatmap.shape
        mid_y, mid_x = h // 2, w // 2

        # Quadrant indices: Superior (top half), Inferior (bottom half), Left, Right
        quadrants = {
            "Superior-Left": float(np.sum(heatmap[:mid_y, :mid_x])),
            "Superior-Right": float(np.sum(heatmap[:mid_y, mid_x:])),
            "Inferior-Left": float(np.sum(heatmap[mid_y:, :mid_x])),
            "Inferior-Right": float(np.sum(heatmap[mid_y:, mid_x:])),
        }

        total_sum = sum(quadrants.values())
        if total_sum > 0:
            quadrant_shares = {k: v / total_sum for k, v in quadrants.items()}
        else:
            quadrant_shares = {k: 0.25 for k in quadrants.keys()}

        # Find the primary quadrant (the one with the largest share)
        primary_quadrant = max(quadrant_shares, key=quadrant_shares.get)
        primary_percentage = quadrant_shares[primary_quadrant]

        # 4. Calculate Overlap ratio with UNeXt segmentation tumor mask
        overlap_pct = 0.0
        if tumor_mask is not None and np.sum(tumor_mask) > 0:
            # Resize heatmap to match tumor mask shape
            h_mask, w_mask = tumor_mask.shape[:2]
            heatmap_resized = cv2.resize(
                heatmap, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR
            )

            # Define high-attention mask (thresholded at 0.5)
            high_attention_mask = heatmap_resized > 0.5
            tumor_binary_mask = tumor_mask > 0

            # Compute intersection and total active attention area
            intersection = np.logical_and(high_attention_mask, tumor_binary_mask).sum()
            attention_total = high_attention_mask.sum()

            if attention_total > 0:
                overlap_pct = float(intersection) / float(attention_total)

        # 5. Compile Natural Language Explanation Text
        method_display = (
            "Grad-CAM"
            if method.lower() == "gradcam"
            else "Grad-CAM++"
            if method.lower() in ["gradcam_plus_plus", "gradcam++"]
            else "EigenCAM"
        )

        explanation_text = (
            f"The classification model (using {method_display}) focuses its diagnostic attention "
            f"primarily on the {primary_quadrant} aspect of the scan, which accounts for "
            f"{primary_percentage:.1%} of the explanation weights."
        )

        if tumor_mask is not None and np.sum(tumor_mask) > 0:
            if overlap_pct >= 0.5:
                explanation_text += (
                    f" There is high spatial overlap ({overlap_pct:.1%}) between the model's diagnostic focus "
                    f"and the segmented lesion boundary, indicating that classification logits are strongly aligned "
                    f"with the physical tumor mass."
                )
            elif overlap_pct >= 0.1:
                explanation_text += (
                    f" There is moderate spatial overlap ({overlap_pct:.1%}) with the segmented tumor mask, "
                    f"suggesting the model is focusing on the lesion but also capturing surrounding structural context."
                )
            else:
                explanation_text += (
                    f" There is low spatial overlap ({overlap_pct:.1%}) between the model's classification attention "
                    f"and the segmented tumor, indicating it may be basing the decision on surrounding tissue or artifacts."
                )
        else:
            explanation_text += " No active tumor mass was detected or provided for spatial overlap analysis."

        metadata = {
            "quadrant_raw_sums": quadrants,
            "heatmap_size": [h, w],
            "overlap_calculated": tumor_mask is not None,
        }

        return ExplanationResult(
            method=method,
            target_class=target_class,
            class_name=class_name,
            heatmap=heatmap,
            explanation_text=explanation_text,
            overlap_percentage=overlap_pct,
            quadrant_attention=quadrant_shares,
            metadata=metadata,
        )
