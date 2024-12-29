# python imports
import os
import io
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Tuple

# installed imports
import magic
from cryptography.fernet import Fernet
from PIL import Image
from werkzeug.utils import secure_filename

# local imports
from ..models import PatientImage, Case


class SecureImageHandler:
    def __init__(self, storage_path: str = os.getenv("UPLOADS")):
        """Initialize the secure image handler.

        Args:
            storage_path: Base path for storing encrypted images
        """

        self.key = Fernet.generate_key()

        self.cipher_suite = Fernet(self.key)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Configure allowed file types
        self.ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def validate_image(self, image_file) -> Tuple[bool, Optional[str]]:
        """Validate image file for type and size.

        Args:
            image_file: File object to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        image_file.seek(0, os.SEEK_END)
        size = image_file.tell()
        image_file.seek(0)

        if size > self.MAX_FILE_SIZE:
            return False, "File size exceeds maximum limit"

        # Check file type
        file_type = magic.from_buffer(image_file.read(1024), mime=True)
        image_file.seek(0)

        if not file_type.startswith("image/"):
            return False, "File is not an image"

        # Verify file extension
        filename = secure_filename(image_file.filename)
        if not "." in filename:
            return False, "No file extension"

        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File extension {ext} not allowed"

        return True, None

    def generate_file_hash(self, file_data: bytes) -> str:
        """Generate SHA-256 hash of file data.

        Args:
            file_data: Bytes of file to hash

        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(file_data).hexdigest()

    def store_image(self, image_file, case_id: str, user_id: str) -> dict:
        """Store image securely with encryption.

        Args:
            image_file: File object to store
            case_id: Associated case ID
            user_id: ID of user uploading image

        Returns:
            Dict containing stored image metadata
        """
        # Validate image
        is_valid, error = self.validate_image(image_file)
        if not is_valid:
            raise ValueError(f"Invalid image: {error}")

        # Generate unique filename
        filename = f"{uuid.uuid4()}.enc"

        # Process and encrypt image
        with Image.open(image_file) as img:
            # Strip EXIF data
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)

            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image_without_exif.save(img_byte_arr, format=img.format)
            img_byte_arr = img_byte_arr.getvalue()

            # Generate content hash
            content_hash = self.generate_file_hash(img_byte_arr)

            # Encrypt
            encrypted_data = self.cipher_suite.encrypt(img_byte_arr)

            # Store encrypted file
            file_path = self.storage_path / filename
            with open(file_path, "wb") as f:
                f.write(encrypted_data)

            # Store metadata in database
            metadata = self._store_image_metadata(
                case_id=case_id,
                filename=filename,
                content_hash=content_hash,
                user_id=user_id,
            )

            return metadata

    def retrieve_image(self, image_id: str, user_id: str) -> Optional[io.BytesIO]:
        """Retrieve and decrypt image.

        Args:
            image_id: ID of image to retrieve
            user_id: ID of user requesting image

        Returns:
            BytesIO object containing decrypted image
        """
        # Verify access
        if not self._verify_access(image_id, user_id):
            raise PermissionError("Unauthorized access to image")

        # Get filename from database
        filename = self._get_filename(image_id)
        if not filename:
            raise FileNotFoundError("Image not found")

        # Read and decrypt file
        file_path = self.storage_path / filename
        with open(file_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = self.cipher_suite.decrypt(encrypted_data)

        return io.BytesIO(decrypted_data)

    def _store_image_metadata(
        self, case_id: str, filename: str, content_hash: str, user_id: str
    ) -> dict:
        """Store image metadata in database."""
        try:
            patient_image = PatientImage(
                case_id=case_id,
                user_id=user_id,
                filename=filename,
                encryption_key=self.key[:10],
                content_hash=content_hash,
            )
            patient_image.insert()
        except Exception as e:
            raise RuntimeError(f"Database error: {str(e)}")

    def _verify_access(self, image_id: str, user_id: str) -> bool:
        """Verify user has access to image."""
        result = (
            PatientImage.query.filter_by(user_id=user_id, is_deleted=False)
            .join(Case)
            .filter_by(user_id=user_id)
            .first()
        )

        return bool(result)

    def _get_filename(self, image_id: str) -> Optional[str]:
        """Get filename from database."""
        result = PatientImage.query.get(image_id)

        return result.filename if result else None
