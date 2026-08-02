import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional
from classification.domain.interfaces import IExplainabilityService


class GradCAMService(IExplainabilityService):
    """Grad-CAM service for generating explanation heatmaps for a PyTorch model.

    This service registers hooks to capture the activations and gradients
    of the target convolutional layer during forward and backward passes.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ) -> None:
        """Initializes the Grad-CAM service.

        Args:
            model: The classification neural network model.
            target_layer: The target convolutional layer (e.g. model.backbone.features[8]).
            device: The device to run computations on.
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

    def _save_activation(
        self, module: nn.Module, input_args: Any, output: torch.Tensor
    ) -> None:
        """Forward hook callback to capture layer activations.

        Args:
            module: The layer module being hooked.
            input_args: Inputs to the module.
            output: Outputs from the module (activations).
        """
        self.activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: Any, grad_output: torch.Tensor
    ) -> None:
        """Backward hook callback to capture gradients of score w.r.t layer activations.

        Args:
            module: The layer module being hooked.
            grad_input: Gradients of loss w.r.t inputs.
            grad_output: Gradients of loss w.r.t outputs.
        """
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self, image_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """Generates a normalized heatmap (0.0 to 1.0) using Grad-CAM.

        Args:
            image_tensor: The input image tensor of shape (C, H, W) or (1, C, H, W).
            target_class: The class index to generate explanation for.

        Returns:
            A 2D numpy array representing the normalized heatmap.
        """
        # Ensure image has a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.clone().to(self.device)
        image_tensor.requires_grad = True

        self.model.eval()

        # Placeholders for hooks
        self.activations = None
        self.gradients = None

        # Register hooks
        handles = []
        try:
            handles.append(
                self.target_layer.register_forward_hook(self._save_activation)
            )
            # Register backward hook. Use register_full_backward_hook if available.
            try:
                handles.append(
                    self.target_layer.register_full_backward_hook(self._save_gradient)
                )
            except AttributeError:
                handles.append(
                    self.target_layer.register_backward_hook(self._save_gradient)
                )

            # Forward pass
            outputs = self.model(image_tensor)

            # Reset gradients
            self.model.zero_grad()

            # Ensure target_class is valid
            num_classes = outputs.shape[1]
            if target_class < 0 or target_class >= num_classes:
                raise ValueError(
                    f"Target class {target_class} is out of bounds for model with {num_classes} classes."
                )

            # Score for the target class
            score = outputs[0, target_class]

            # Backward pass to compute gradients
            score.backward()

        except Exception as e:
            raise RuntimeError(f"Error during Grad-CAM backpropagation: {e}") from e
        finally:
            # Always remove hooks to prevent memory leaks
            for handle in handles:
                handle.remove()

        if self.activations is None:
            raise RuntimeError(
                "Forward hook failed to capture activations. Check if the target layer is correct."
            )
        if self.gradients is None:
            raise RuntimeError(
                "Backward hook failed to capture gradients. Check if backpropagation succeeded."
            )

        # Pool the gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        weighted_activations = weights * self.activations
        cam = torch.sum(weighted_activations, dim=1).squeeze(0)

        # Pass through ReLU
        cam = torch.clamp(cam, min=0.0)

        # Move to CPU for numpy operations
        cam_np = cam.cpu().numpy()

        # Handle edge case where CAM is all zeros or flat to avoid division by zero
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        denominator = max_val - min_val
        if denominator > 1e-8:
            heatmap = (cam_np - min_val) / denominator
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap
