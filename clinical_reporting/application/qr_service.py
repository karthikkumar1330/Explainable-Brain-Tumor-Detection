import os
import qrcode
import logging

class QRGenerationError(Exception):
    """Exception raised when QR code generation fails."""
    pass

class QRGeneratorService:
    """Service to handle generation of secure verification QR codes."""

    def __init__(self, verification_base_url: str = "http://localhost:5000/verify") -> None:
        """Initializes the QR generator service with a configurable base URL."""
        self.verification_base_url = verification_base_url.rstrip('/')
        self.logger = logging.getLogger("qr_generator_service")

    def generate_verification_url(self, token: str) -> str:
        """Constructs the verification URL for a given token."""
        return f"{self.verification_base_url}/{token}"

    def generate_qr_code_image(self, token: str, output_path: str, box_size: int = 10, border: int = 4) -> str:
        """Generates a high-quality QR code image of the verification URL.

        Args:
            token: Secure token identifying the report version.
            output_path: Path where the generated image should be saved.
            box_size: Size of each QR code box (pixel scale).
            border: Width of the white border.

        Returns:
            The absolute path of the generated QR code image.

        Raises:
            QRGenerationError: If the QR code image cannot be created or saved.
        """
        try:
            url = self.generate_verification_url(token)
            self.logger.info(f"Generating QR code image for token. Target URL: {url}")

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=box_size,
                border=border,
            )
            qr.add_data(url)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Ensure parent directories exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            img.save(output_path)
            
            self.logger.info(f"QR code successfully generated and saved to: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to generate QR code: {e}")
            raise QRGenerationError(f"Failed to generate QR code: {e}")
