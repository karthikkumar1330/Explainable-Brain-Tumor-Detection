import logging
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from classification.domain.entities import BrainTumorClass, PredictionResult
from classification.domain.interfaces import IModelAdapter, IExplainabilityService


class TrainModelUseCase:
    """Use case to manage and execute the training loop for the classification model."""

    def __init__(
        self, model_adapter: IModelAdapter, logger: logging.Logger
    ) -> None:
        """Initializes the train model use case.

        Args:
            model_adapter: Implementation of the model adapter.
            logger: The logger instance for printing/writing updates.
        """
        self.model_adapter = model_adapter
        self.logger = logger

    def execute(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Runs the training and validation loops across epochs.

        Saves the model checkpoint when a new highest validation F1 score is achieved.

        Args:
            train_loader: The data loader for training images.
            val_loader: The data loader for validation images.
            epochs: Total training epochs.
            checkpoint_path: Filepath where the best model checkpoint should be saved.

        Returns:
            A dictionary containing historical training/validation metrics.
        """
        self.logger.info("Starting model training pipeline...")
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        best_val_f1 = -1.0

        for epoch in range(1, epochs + 1):
            self.logger.info(f"Epoch {epoch}/{epochs}")

            # Training epoch
            try:
                train_loss = self.model_adapter.train_epoch(train_loader)
                self.logger.info(f"  Train Loss: {train_loss:.4f}")
                history["train_loss"].append(train_loss)
            except Exception as e:
                self.logger.error(f"Error during training epoch {epoch}: {e}")
                raise e

            # Validation step
            try:
                val_loss, metrics = self.model_adapter.evaluate(val_loader)
                val_acc = metrics.get("accuracy", 0.0)
                val_f1 = metrics.get("f1_score", 0.0)

                self.logger.info(
                    f"  Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val F1-Score: {val_f1:.4f}"
                )

                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                history["val_f1"].append(val_f1)

                # Save best checkpoint
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.model_adapter.save(checkpoint_path)
                    self.logger.info(
                        f"  --> Saved new best checkpoint to {checkpoint_path} "
                        f"(Best F1-Score: {best_val_f1:.4f})"
                    )
            except Exception as e:
                self.logger.error(f"Error during validation epoch {epoch}: {e}")
                raise e

        self.logger.info("Model training pipeline complete.")
        return history


class EvaluateModelUseCase:
    """Use case to evaluate model performance on a test/validation dataset."""

    def __init__(
        self, model_adapter: IModelAdapter, logger: logging.Logger
    ) -> None:
        """Initializes the evaluate model use case.

        Args:
            model_adapter: Implementation of the model adapter.
            logger: The logger instance.
        """
        self.model_adapter = model_adapter
        self.logger = logger

    def execute(self, dataloader: Any) -> Tuple[float, Dict[str, float]]:
        """Evaluates the model on the provided data loader.

        Args:
            dataloader: Data loader for evaluation.

        Returns:
            A tuple of (average loss, metrics dictionary).
        """
        self.logger.info("Running model evaluation...")
        try:
            val_loss, metrics = self.model_adapter.evaluate(dataloader)
            self.logger.info(f"Evaluation Loss: {val_loss:.4f}")
            for name, value in metrics.items():
                self.logger.info(f"  {name}: {value:.4f}")
            return val_loss, metrics
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise e


class PredictUseCase:
    """Use case to predict class categories and confidence scores for single images."""

    def __init__(self, model_adapter: IModelAdapter) -> None:
        """Initializes the predict use case.

        Args:
            model_adapter: Implementation of the model adapter.
        """
        self.model_adapter = model_adapter

    def execute(self, image_tensor: Any) -> PredictionResult:
        """Runs inference on a preprocessed image tensor.

        Args:
            image_tensor: Normalized image tensor (C, H, W).

        Returns:
            A PredictionResult entity.
        """
        label, confidence, probs = self.model_adapter.predict(image_tensor)
        class_name = BrainTumorClass.get_name_by_value(label)

        return PredictionResult(
            label=label,
            class_name=class_name,
            confidence_score=confidence,
            probabilities=probs,
        )


class ExplainPredictionUseCase:
    """Use case to run prediction and generate explanation heatmaps using Grad-CAM."""

    def __init__(
        self,
        predict_use_case: PredictUseCase,
        explain_service: IExplainabilityService,
        logger: logging.Logger,
    ) -> None:
        """Initializes the explain prediction use case.

        Args:
            predict_use_case: Use case for standard prediction.
            explain_service: Service to generate explainability maps.
            logger: The logger instance.
        """
        self.predict_use_case = predict_use_case
        self.explain_service = explain_service
        self.logger = logger

    def execute(
        self, image_tensor: Any, target_class: Optional[int] = None
    ) -> Tuple[PredictionResult, np.ndarray]:
        """Runs prediction and generates explainability map for the image tensor.

        If target_class is None, the predicted class will be used.

        Args:
            image_tensor: Normalized image tensor (C, H, W) or (1, C, H, W).
            target_class: Optional target class index.

        Returns:
            A tuple containing:
                - PredictionResult: The predictions.
                - np.ndarray: The 2D normalized heatmap (0.0 to 1.0).
        """
        self.logger.info("Executing explainable prediction...")
        # 1. Run prediction
        try:
            result = self.predict_use_case.execute(image_tensor)
            self.logger.info(
                f"Prediction successful. Class: {result.class_name} "
                f"(confidence: {result.confidence_score:.4f})"
            )
        except Exception as e:
            self.logger.error(
                f"Error during prediction inside explain use case: {e}"
            )
            raise e

        # 2. Determine target class index
        actual_target = target_class if target_class is not None else result.label
        self.logger.info(f"Generating Grad-CAM for class index: {actual_target}")

        # 3. Generate heatmap
        try:
            heatmap = self.explain_service.generate_heatmap(
                image_tensor, actual_target
            )
            self.logger.info("Grad-CAM heatmap generated successfully.")
            return result, heatmap
        except Exception as e:
            self.logger.error(f"Error during Grad-CAM generation: {e}")
            raise e
