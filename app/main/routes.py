# installed imports
from flask import Blueprint, render_template
from flask_login import login_required, current_user

# local imports
from .. import logger
from ..models import Case

main = Blueprint("main", __name__)

# Index ---------------------------
@main.get("/")
def index():
    return render_template("main/index.html")


# User Account -------------------------------
@main.get("/profile")
@login_required
def account():
    return render_template("main/profile.html", title="Profile")


# Terms of Service -------------------------------
@main.get("/terms-of-service")
def terms_of_service():
    return render_template("main/tos.html", title="Terms of Service")


# Privacy Policy ---------------------------------
@main.get("/privacy-policy")
def privacy_policy():
    return render_template("main/privacy.html", title="Pricacy Policy")


# Dashboard -------------------------------------
@main.get("/cases")
@login_required
def profile():
    cases = Case.query.filter(Case.user_id == current_user.id).all()
    return render_template("main/cases.html", cases=cases)


# Add Case ------------------------------------
# TODO
