# installed imports
import traceback
from flask_login import login_required, current_user
from flask import Blueprint, render_template, abort, flash, redirect, url_for

# local imports
from ..models import Case, User
from .. import logger, csrf

main = Blueprint("main", __name__)

# Index ---------------------------
@main.get("/")
@main.get("/case")
@login_required
def index():
    return render_template("main/index.html", cases=current_user.cases)


# User Account -------------------------------
@main.get("/profile")
@login_required
def account():
    return render_template("main/profile.html", title="Profile")


# Terms of Service -------------------------------
@main.get("/terms-of-service")
def terms_of_service():
    return render_template("main/tos.html", title="Terms of Service")


# About ---------------------------------
@main.get("/about")
def about():
    return render_template("main/about.html", title="About")


# Privacy Policy ---------------------------------
@main.get("/privacy-policy")
def privacy_policy():
    return render_template("main/privacy.html", title="Pricacy Policy")


# Case View ------------------------------------
@main.get("/case/<uid>")
@login_required
def show_case(uid):
    return render_template(
        "main/case.html",
        case=Case.query.filter_by(user_id=current_user.id, uid=uid).one_or_404(),
    )


# Delete Case -------------------------------
@main.get("/case/delete/<uid>")
@login_required
def delete_case(uid):
    try:
        from ..modules.secure_image_handler import SecureImageHandler

        handler = SecureImageHandler()
        get_case = Case.query.filter_by(user_id=current_user.id, uid=uid).one_or_404()
        for image in get_case.patient_images:
            handler.delete_image(image.id, current_user.id)
        get_case.delete()
        flash(f"Delete successful", "success")
        return redirect(url_for("main.index"))
    except:
        logger.error(traceback.format_exc())
        abort(500)
