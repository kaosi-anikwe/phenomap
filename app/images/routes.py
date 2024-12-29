# installed imports
from flask_login import login_required, current_user
from flask import Blueprint, request, jsonify, send_file

# local imports
from ..modules.secure_image_handler import SecureImageHandler

images = Blueprint("images", __name__)
image_handler = SecureImageHandler()


@images.route("/upload", methods=["POST"])
@login_required
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    case_id = request.form.get("case_id")

    try:
        metadata = image_handler.store_image(image, case_id, current_user.id)
        return jsonify(metadata), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Upload failed"}), 500


@images.route("/<image_id>", methods=["GET"])
@login_required
def get_image(image_id):
    try:
        image_data = image_handler.retrieve_image(image_id, current_user.id)
        return send_file(
            image_data,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=f"image_{image_id}.jpg",
        )

    except PermissionError:
        return jsonify({"error": "Unauthorized"}), 403
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": "Failed to retrieve image"}), 500
