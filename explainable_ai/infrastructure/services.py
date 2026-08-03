import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional
from explainable_ai.domain.interfaces import IXAIEngine


class PyTorchXAIEngine(IXAIEngine):
    """Explainable AI (XAI) 2.0 Engine.

    Implements Grad-CAM, Grad-CAM++, and EigenCAM methods for PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ) -> None:
        """Initializes the PyTorch XAI Engine.

        Args:
            model: The neural network classification model (e.g. EfficientNetB0Model).
            target_layer: The target convolutional layer to hook.
            device: Computing device.
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.active_method: str = "gradcam"

    def set_method(self, method: str) -> None:
        """Sets the active explainability method."""
        method_lower = method.lower()
        if method_lower not in ["gradcam", "gradcam_plus_plus", "gradcam++", "eigencam"]:
            raise ValueError(f"Unsupported XAI method: {method}")
        self.active_method = method_lower

    def _save_activation(
        self, module: nn.Module, input_args: Any, output: torch.Tensor
    ) -> None:
        """Forward hook callback to capture activations."""
        self.activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: Any, grad_output: torch.Tensor
    ) -> None:
        """Backward hook callback to capture gradients."""
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def _run_forward_backward(self, image_tensor: torch.Tensor, target_class: int) -> None:
        """Helper to run the forward and backward passes with registered hooks."""
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.clone().to(self.device)
        image_tensor.requires_grad = True

        self.model.eval()

        # Reset hooks placeholders
        self.activations = None
        self.gradients = None

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

            num_classes = outputs.shape[1]
            if target_class < 0 or target_class >= num_classes:
                raise ValueError(
                    f"Target class {target_class} is out of bounds for model with {num_classes} classes."
                )

            # Score for target class
            score = outputs[0, target_class]

            # Backward pass to compute gradients
            score.backward()

        except Exception as e:
            raise RuntimeError(f"Error during XAI backpropagation: {e}") from e
        finally:
            # Always remove hooks to prevent memory leaks
            for handle in handles:
                handle.remove()

    def _generate_gradcam(self) -> np.ndarray:
        """Computes standard Grad-CAM heatmap."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations/gradients are missing. Ensure backpropagation was run.")
        
        # Mean pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = weights * self.activations
        cam = torch.sum(weighted_activations, dim=1).squeeze(0)
        
        # Apply ReLU
        cam = torch.clamp(cam, min=0.0)
        return cam.cpu().numpy()

    def _generate_gradcam_plus_plus(self) -> np.ndarray:
        """Computes Grad-CAM++ heatmap using localized first-derivative approximation."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations/gradients are missing. Ensure backpropagation was run.")
        
        grads = self.gradients
        acts = self.activations
        
        # Calculate localized alpha coefficients
        grads_positive = torch.clamp(grads, min=0.0)
        grads_power2 = grads ** 2
        grads_power3 = grads ** 3
        
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        denominator = 2.0 * grads_power2 + sum_acts * grads_power3
        
        eps = 1e-8
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator) * eps)
        
        alphas = grads_power2 / denominator
        weights = torch.sum(alphas * grads_positive, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        weighted_activations = weights * acts
        cam = torch.sum(weighted_activations, dim=1).squeeze(0)
        
        # Apply ReLU
        cam = torch.clamp(cam, min=0.0)
        return cam.cpu().numpy()

    def _generate_eigencam(self) -> np.ndarray:
        """Computes gradient-free EigenCAM heatmap using SVD on activations."""
        if self.activations is None:
            raise RuntimeError("Activations are missing. Ensure forward pass was run.")
        
        acts = self.activations.squeeze(0).cpu().numpy()  # Shape: (C, H, W)
        C, H, W = acts.shape
        X = acts.transpose(1, 2, 0).reshape(H * W, C)
        
        # Center channels
        X_centered = X - np.mean(X, axis=0)
        
        # Singular Value Decomposition (SVD)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            cam = U[:, 0].reshape(H, W)
        except Exception:
            # Fallback to mean activation across channels if SVD fails to converge
            cam = np.mean(acts, axis=0)

        # Standardize sign: ensure positive correlation with mean activations
        mean_acts = np.mean(acts, axis=0)
        if np.sum(cam * mean_acts) < 0:
            cam = -cam
            
        return cam

    def generate_explanation(
        self, image_tensor: Any, target_class: int, method: str = "gradcam"
    ) -> np.ndarray:
        """Generates heatmap for target class using specified XAI method."""
        method_lower = method.lower()
        
        # EigenCAM does not strictly require gradients, but running standard forward-backward
        # is safe, unified, and populates self.activations correctly.
        self._run_forward_backward(image_tensor, target_class)

        if method_lower == "gradcam":
            cam_np = self._generate_gradcam()
        elif method_lower in ["gradcam_plus_plus", "gradcam++"]:
            cam_np = self._generate_gradcam_plus_plus()
        elif method_lower == "eigencam":
            cam_np = self._generate_eigencam()
        else:
            raise ValueError(f"Unsupported XAI method: {method}")

        # Normalize heatmap to [0, 1]
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        denominator = max_val - min_val
        if denominator > 1e-8:
            heatmap = (cam_np - min_val) / denominator
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap

    def generate_heatmap(
        self, image_tensor: Any, target_class: int
    ) -> np.ndarray:
        """Backward-compatible method mapping standard requests to the active method."""
        return self.generate_explanation(image_tensor, target_class, method=self.active_method)
