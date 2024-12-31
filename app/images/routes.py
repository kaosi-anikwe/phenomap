# python imports
import traceback

# installed imports
from flask_login import login_required, current_user
from flask import Blueprint, request, jsonify, send_file

# local imports
from .. import logger
from ..models import Case, PatientImage
from ..modules.secure_image_handler import SecureImageHandler

images = Blueprint("images", __name__)


# Upload Image -----------------------------
@images.post("/upload")
@login_required
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    case_id = request.form.get("case_id")

    try:
        image_handler = SecureImageHandler()
        get_case = Case.query.get(case_id)
        is_default = len(get_case.patient_images) == 0
        metadata = image_handler.store_image(
            image, case_id, current_user.id, is_default
        )
        return jsonify(metadata), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Upload failed"}), 500


# Update Image ------------------------------------------
@images.post("/update")
@login_required
def update_image():
    try:
        data = request.get_json()
        logger.info(f"Updating Image with data: {data}")
        image = PatientImage.query.get(data.get("id"))
        if image:
            if data.get("value"):
                if data.get("field") == "is_default":
                    # Clear other default
                    img = PatientImage.query.filter(
                        PatientImage.case_id == data.get("case_id"),
                        PatientImage.is_default == True,
                    ).one()
                    img.is_default = False
                    img.update()
                setattr(image, data.get("field"), data.get("value"))
                image.update()
            return jsonify(success=True)
        else:
            return jsonify(success=False, error="Image record not found"), 404
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(success=False, error=str(e)), 500


# GET / DELETE Image -----------------------------------
@images.route("/<image_id>", methods=["GET", "DELETE"])
@login_required
def get_image(image_id):
    try:
        image = PatientImage.query.filter(PatientImage.id == image_id).first()
        if not image:
            raise FileNotFoundError("Image record not found")
        image_handler = SecureImageHandler(key=image.encryption_key)
        image_data = image_handler.retrieve_image(image_id, current_user.id)
        logger.info(f"Retrieving image #{image_id} for user #{current_user.id}")
        if request.method == "GET":
            logger.info("Sending image")
            return send_file(
                image_data,
                mimetype="image/jpeg",
                as_attachment=False,
                download_name=f"image_{image_id}.jpg",
            )
        if request.method == "DELETE":
            logger.info(f"Deleting image")
            success = image_handler.delete_image(image_id, current_user.id)
            if success:
                return jsonify(success=True), 200
            else:
                return jsonify(success=False, error="Failed to delete image"), 500
    except PermissionError:
        return jsonify({"error": "Unauthorized"}), 403
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to retrieve image"}), 500
