import unittest
import torch
import numpy as np
from research_framework.domain.entities import ModelProfile, ModelPrediction, EnsembleResult
from research_framework.application.registry import ModelRegistry
from research_framework.application.ensemble import EnsembleEngine


class TestMultiModelResearchFramework(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ModelRegistry()
        self.engine = EnsembleEngine()
        self.image_tensor = torch.zeros((1, 3, 224, 224))

    def test_model_registry_profiles(self):
        """Verifies that all three classification profiles are registered."""
        self.assertIn("efficientnet_b0", self.registry.profiles)
        self.assertIn("resnet18", self.registry.profiles)
        self.assertIn("mobilenet_v3", self.registry.profiles)
        
        prof = self.registry.profiles["resnet18"]
        self.assertEqual(prof.architecture, "ResNet-18")
        self.assertFalse(prof.is_production)

    def test_model_instantiation(self):
        """Verifies registry can load research models (resnet18/mobilenet_v3) without error."""
        resnet = self.registry.get_loaded_model("resnet18", device="cpu")
        mobilenet = self.registry.get_loaded_model("mobilenet_v3", device="cpu")
        self.assertIsNotNone(resnet)
        self.assertIsNotNone(mobilenet)

    def test_predict_all_returns_predictions(self):
        """Verifies predict_all runs inferences and yields individual model predictions."""
        predictions = self.registry.predict_all(self.image_tensor, device="cpu")
        self.assertEqual(len(predictions), 3)
        for p in predictions:
            self.assertIn(p.model_name, ["efficientnet_b0", "resnet18", "mobilenet_v3"])
            self.assertIn(p.predicted_class, ["Glioma", "Meningioma", "No Tumor", "Pituitary"])
            self.assertGreater(p.confidence, 0.0)
            self.assertGreater(p.runtime_sec, 0.0)

    def test_ensemble_soft_voting_calculation(self):
        """Tests that soft voting averages probability values and selects maximum."""
        p1 = ModelPrediction(
            model_name="efficientnet_b0",
            predicted_class="Glioma",
            confidence=0.80,
            probabilities={"Glioma": 0.80, "Meningioma": 0.10, "No Tumor": 0.05, "Pituitary": 0.05},
            runtime_sec=0.05
        )
        p2 = ModelPrediction(
            model_name="resnet18",
            predicted_class="Meningioma",
            confidence=0.60,
            probabilities={"Glioma": 0.30, "Meningioma": 0.60, "No Tumor": 0.05, "Pituitary": 0.05},
            runtime_sec=0.04
        )
        
        res = self.engine.compute_ensemble([p1, p2])
        self.assertIsInstance(res, EnsembleResult)
        
        # Averaged probabilities:
        # Glioma: (0.80 + 0.30)/2 = 0.55
        # Meningioma: (0.10 + 0.60)/2 = 0.35
        # No Tumor: 0.05
        # Pituitary: 0.05
        # Maximizer is Glioma (0.55)
        self.assertEqual(res.predicted_class, "Glioma")
        self.assertAlmostEqual(res.confidence, 0.55)
        self.assertEqual(res.probabilities["Glioma"], 0.55)
        self.assertEqual(res.probabilities["Meningioma"], 0.35)

    def test_calculate_agreement_indicators(self):
        """Verifies calculation of Cosine Similarity and JSD agreement status."""
        p_prod = ModelPrediction(
            model_name="efficientnet_b0",
            predicted_class="Glioma",
            confidence=0.90,
            probabilities={"Glioma": 0.90, "Meningioma": 0.04, "No Tumor": 0.03, "Pituitary": 0.03},
            runtime_sec=0.05
        )
        p_res = ModelPrediction(
            model_name="resnet18",
            predicted_class="Glioma",
            confidence=0.88,
            probabilities={"Glioma": 0.88, "Meningioma": 0.04, "No Tumor": 0.04, "Pituitary": 0.04},
            runtime_sec=0.04
        )
        
        metrics = self.engine.calculate_agreement(p_prod, [p_prod, p_res])
        self.assertGreater(metrics.cosine_similarity, 0.98)
        self.assertLess(metrics.jensen_shannon_divergence, 0.01)
        self.assertEqual(metrics.level, "HIGH CONSENSUS")
