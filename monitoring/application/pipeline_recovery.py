import torch
import logging
from typing import Callable, Any, Tuple, List, Optional


class PipelineExecutionRecovery:
    """Service handling automatic exception retries, device fallback (CUDA to CPU), and graceful degradation."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("pipeline_recovery")

    def run_inference_with_fallback(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, str, List[str]]:
        """Runs model inference on preferred device, falling back to CPU if execution fails.

        Args:
            model: PyTorch model.
            input_tensor: Input tensor on preferred device.
            device: Preferred execution device.

        Returns:
            A tuple of (output_tensor, active_device_name, warnings_list).
        """
        warnings: List[str] = []
        active_device = str(device)

        # Ensure model is in eval mode and correct device
        model.eval()
        try:
            # Attempt preferred device run
            with torch.inference_mode():
                output = model(input_tensor.to(device))
                return output, active_device, warnings
        except Exception as e:
            self.logger.warning(f"Inference failed on device {device}: {e}. Retrying with CPU fallback...")
            warnings.append(f"Auto-recovery warning: Execution failed on {device}. Fell back to CPU fallback mode.")
            
            try:
                # Force CPU fallback retry
                cpu_device = torch.device("cpu")
                model_cpu = model.to(cpu_device)
                tensor_cpu = input_tensor.to(cpu_device)
                
                with torch.inference_mode():
                    output = model_cpu(tensor_cpu)
                return output, "cpu", warnings
            except Exception as cpu_err:
                self.logger.critical(f"Critical execution failure: CPU fallback retry also failed: {cpu_err}")
                raise cpu_err

    def execute_graceful_stage(
        self,
        stage_name: str,
        stage_fn: Callable[[], Any],
        default_fallback_value: Any
    ) -> Tuple[Any, List[str]]:
        """Wraps auxiliary stages with recovery. If they fail, returns default fallback and logs warnings.

        Args:
            stage_name: Name of pipeline step (e.g. "Grad-CAM", "Tumor Statistics").
            stage_fn: Function executing the logic.
            default_fallback_value: Value to return if step fails.

        Returns:
            A tuple of (outcome_value, warnings_list).
        """
        warnings: List[str] = []
        try:
            result = stage_fn()
            return result, warnings
        except Exception as e:
            self.logger.error(f"Graceful degradation: Auxiliary pipeline stage '{stage_name}' failed: {e}")
            warnings.append(f"Graceful degradation: Optional stage '{stage_name}' failed ({e}). Default placeholders applied.")
            return default_fallback_value, warnings
