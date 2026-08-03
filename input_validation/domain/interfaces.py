from abc import ABC, abstractmethod
from input_validation.domain.entities import ValidationScorecard


class IMriValidator(ABC):
    """Interface defining operations for validating uploaded MRI scan files and structures."""

    @abstractmethod
    def validate_file(
        self,
        filepath: str,
        file_bytes: bytes,
        filename: str
    ) -> ValidationScorecard:
        """Runs the validation suite on raw scan file bytes and metadata.

        Args:
            filepath: Path where the scan file is temporarily saved.
            file_bytes: Raw binary bytes of the file.
            filename: Original filename of the upload.

        Returns:
            A ValidationScorecard containing verification results.
        """
        pass
