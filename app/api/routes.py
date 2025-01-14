# python imports
import os
import traceback
import threading
from datetime import datetime

# installed imports
from flask import Blueprint, request, jsonify
from flask_login import current_user, login_required

# local imports
from .. import logger
from ..modules.prediction import ClassificationManager
from ..modules.secure_image_handler import SecureImageHandler
from ..models import (
    Case,
    CaseNote,
    Syndrome,
    Prediction,
    PatientImage,
    CasePredictionDiagnosis,
)


api = Blueprint("api", __name__)
classification_manager = ClassificationManager()


# Get Cases -----------------------------
@api.get("/")
@login_required
def get_all_cases():
    return jsonify(
        data=[
            case.parse()
            for case in Case.query.filter_by(user_id=current_user.id)
            .order_by(Case.id.desc())
            .all()
        ]
    )


# Delete Case --------------------------------
@api.delete("/<uid>")
@login_required
def delete_case(uid):
    try:
        handler = SecureImageHandler()
        get_case = Case.query.filter_by(user_id=current_user.id, uid=uid).one_or_404()
        for image in get_case.patient_images:
            handler.delete_image(image.id, current_user.id)
        get_case.delete()
        logger.info("Case deleted successfully")
        return jsonify(success=True)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(success=False, message=str(e))


# Case ------------------------------------
@api.post("/update")
@login_required
def update_case():
    try:
        data = request.get_json()
        logger.info(f"Updating case with data: {data}")
        if not data["uid"]:
            # new case
            logger.info("Creating new case")
            new_case = Case(current_user.id)
            new_case.insert()
            logger.info(f"Case created with uid: {new_case.uid}")
            return jsonify(success=True, uid=new_case.uid)

        # Update Case
        get_case = Case.query.filter(
            Case.user_id == current_user.id, Case.uid == data.get("uid")
        ).one_or_none()
        if get_case:
            if data.get("value"):
                if data.get("field") == "dob":
                    logger.info("Updating DoB")
                    date_object = datetime.strptime(
                        data.get("value")[:10], "%Y-%m-%d"
                    ).date()
                    get_case.dob = date_object
                else:
                    logger.info(f"Updating {data.get('field')} to {data.get('value')}")
                    setattr(get_case, data.get("field"), data.get("value"))
                get_case.update()
            return jsonify(success=True, uid=get_case.uid)
        else:
            return jsonify(success=False, error="Case not found"), 404
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(success=False, error=str(e)), 500


# Case Images -------------------------------------------
@api.get("/<case_id>/images")
@login_required
def get_case_images(case_id):
    try:
        result = PatientImage.query.filter(
            PatientImage.case_id == case_id, PatientImage.is_deleted == False
        ).all()
        images = [
            {
                "id": image.id,
                "caseId": image.case_id,
                "type": image.type,
                "description": image.description,
                "date_taken": image.date_taken.isoformat()
                if image.date_taken
                else None,
                "is_default": image.is_default,
            }
            for image in result
        ]

        return jsonify({"images": images})

    except Exception as e:
        print(f"Error fetching images: {str(e)}")
        return jsonify({"error": "Failed to fetch images"}), 500


# Case Notes  GET -------------------------------------------
@api.get("/<case_id>/notes")
@login_required
def get_case_notes(case_id):
    try:
        notes = (
            CaseNote.query.filter(CaseNote.case_id == case_id)
            .order_by(CaseNote.created_at)
            .all()
        )
        return jsonify(
            notes=[
                {
                    "id": note.id,
                    "content": note.content,
                    "created_at": note.created_at.isoformat(),
                    "user": {
                        "id": current_user.id,
                        "name": current_user.display_name(),
                    },
                }
                for note in notes
            ]
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(success=False, message=str(e))


# Case Notes POST --------------------------------------
@api.post("/<case_id>/notes")
@login_required
def add_case_note(case_id):
    try:
        data = request.get_json()
        logger.info(f"Adding note with data: {data}")
        note = CaseNote(data.get("content"), case_id)
        note.insert()
        return jsonify(
            {
                "id": note.id,
                "content": note.content,
                "created_at": note.created_at.isoformat(),
                "user": {
                    "id": current_user.id,
                    "name": current_user.display_name(),
                },
            }
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(success=False, message=str(e))


# Case Prediction ----------------------------------------
@api.route("/<case_id>/prerequisites", methods=["GET"])
@login_required
def check_prerequisites(case_id):
    """Check if all requirements are met for classification."""
    try:
        case = Case.query.get(case_id)
        return jsonify(case.check_classification_prerequisites())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Case Classification -------------------------------------
@api.route("/<case_id>/classify", methods=["POST"])
@login_required
def perform_classification(case_id):
    """Perform syndrome classification for a case."""
    try:
        logger.info(f"Received ClassificationRequest for Case #{case_id}")
        force = request.json.get("force", False)
        if not force:
            if Prediction.query.filter(Prediction.case_id == case_id).first():
                return jsonify({"error": "Classifications already exist"}), 400
        # Verify prerequisites
        prereq_check = check_prerequisites(case_id).get_json()
        if not prereq_check["can_classify"]:
            return (
                jsonify(
                    {
                        "error": "Prerequisites not met",
                        "missing": prereq_check["missing_prerequisites"],
                    }
                ),
                400,
            )
        request_data = classification_manager.create_request(
            case_id=case_id, force=force
        )

        # Start processing in background task
        thrd = threading.Thread(
            target=classification_manager.process_request,
            args=(request_data["id"],),
        )
        thrd.start()

        return jsonify({"request_id": request_data["id"], "status": "pending"})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Case Prediction ---------------------------------------------
@api.route("/<case_id>/predictions", methods=["GET"])
@login_required
def get_predictions(case_id):
    """Get existing predictions for a case."""
    try:
        logger.info(f"Retrieving Predictions for Case #{case_id}")
        results = (
            Prediction.query.join(CasePredictionDiagnosis)
            .filter(
                Prediction.case_id == case_id,
                CasePredictionDiagnosis.prediction_id == Prediction.id,
            )
            .order_by(Prediction.confidence_score.desc())
            .all()
        )
        predictions = [
            {
                "id": row.id,
                "syndrome_name": row.syndrome_name,
                "syndrome_code": row.syndrome_code,
                "confidence_score": float(row.confidence_score),
                "status": row.status,
                "composite_image": row.composite_image,
                "is_removed": row.is_removed,
                "diagnosis_status": {
                    "differential": row.diagnosis.differential,
                    "clinically_diagnosed": row.diagnosis.clinically_diagnosed,
                    "molecularly_diagnosed": row.diagnosis.molecularly_diagnosed,
                },
            }
            for row in results
        ]
        logger.info(f"Returning {len(predictions)} predictions")
        return jsonify({"predictions": predictions})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@api.route("/<case_id>/predictions/<prediction_id>", methods=["GET", "PATCH"])
@login_required
def update_prediction(case_id, prediction_id):
    """Update prediction status (remove/restore)."""
    try:
        prediction = Prediction.query.filter_by(
            id=prediction_id, case_id=case_id
        ).one_or_none()
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        if request.method == "PATCH":
            logger.info(f"Updating prediction with data: {request.json}")
            is_removed = request.json.get("is_removed", False)
            prediction.is_removed = is_removed
            prediction.update()
            return jsonify({"success": True})
        if request.method == "GET":
            logger.info(f"Returning Prediction #{prediction.id}")
            return jsonify(prediction.json())
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Case Classification Request Status ------------------------------
@api.get("/<case_id>/requests/<request_id>/status")
@login_required
def get_classification_status(case_id, request_id):
    logger.info(f"Checking status of Classification Request #{request_id}")
    classifcation_manager = ClassificationManager()
    status = classifcation_manager.get_request_status(request_id)
    logger.info(f"Status is {status['status']}")
    if not status:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(status)


# Syndromes ----------------------------------------
@api.get("/syndrome/<syndrome_code>")
@login_required
def get_syndrome(syndrome_code):
    try:
        logger.info(f"Attempting to get Syndrome with code: {syndrome_code}")
        syndrome = Syndrome.query.filter(Syndrome.code == syndrome_code).one()
        return jsonify(syndrome.json())
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
