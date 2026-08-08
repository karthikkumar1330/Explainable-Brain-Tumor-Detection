import os
import io
from PIL import Image

def validate_and_resize_avatar(file_stream, filename: str, max_size=(256, 256)) -> bytes:
    """
    Validates that the file is a valid image, converts/resizes it to a standard size (e.g. 256x256),
    strips metadata (EXIF), and returns the resized image bytes.
    Raises ValueError with a descriptive message if the image is invalid, corrupt, or has an unsupported format.
    """
    # Enforce safe file extension check
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    if ext not in allowed_exts:
        raise ValueError("Invalid image format. Allowed formats: PNG, JPG, JPEG, GIF, WEBP.")

    try:
        # Load the image using Pillow
        img = Image.open(file_stream)
        
        # Read the pixel data to verify the image is not corrupted or truncated
        img.load()
        
        # Validate format strictly
        detected_format = img.format.upper() if img.format else ""
        if detected_format not in ["PNG", "JPEG", "GIF", "WEBP"]:
            raise ValueError(f"Invalid image format. Detected format {detected_format} is not allowed.")
        
        # Use LANCZOS resampling for high-quality downscaling
        try:
            resample_mode = Image.Resampling.LANCZOS
        except AttributeError:
            resample_mode = Image.LANCZOS
            
        # Resize maintaining aspect ratio
        img.thumbnail(max_size, resample_mode)
        
        # Write to bytes buffer to strip EXIF and other metadata
        out_buf = io.BytesIO()
        img.save(out_buf, format=detected_format)
        
        return out_buf.getvalue()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid image file content. Could not decode image: {str(e)}")
