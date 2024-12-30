# installed imports
from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

# local imports
from .. import logger
from ..models import Case

main = Blueprint("main", __name__)

# Index ---------------------------
@main.get("/")
@login_required
def index():
    cases = Case.query.filter(Case.user_id == current_user.id).all()
    return render_template("main/index.html", cases=cases)


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


# Add Case ------------------------------------
# TODO
