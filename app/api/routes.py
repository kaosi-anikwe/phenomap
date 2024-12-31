# python imports
import traceback
from datetime import datetime

# installed imports
from flask import Blueprint, request, jsonify
from flask_login import current_user, login_required

# local imports
from .. import logger
from ..models import Case, PatientImage, CaseNote
from ..modules.secure_image_handler import SecureImageHandler


api = Blueprint("api", __name__)


# Get Cases -----------------------------
@api.get("/")
@login_required
def get_all_cases():
    return jsonify(
        data=[
            case.parse() for case in Case.query.filter_by(user_id=current_user.id).all()
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


# Case Notes -------------------------------------------
@api.get("/<case_id>/notes")
@login_required
def get_case_notes(case_id):
    try:
        notes = (
            CaseNote.query.filter(CaseNote.case_id == case_id)
            .order_by(CaseNote.created_at.desc())
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
