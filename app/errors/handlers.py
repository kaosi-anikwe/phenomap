from flask import Blueprint, render_template

errors = Blueprint("errors", __name__)


@errors.app_errorhandler(400)
def bad_request(error):
    return render_template("errors/400.html", title="Error 400"), 400


@errors.app_errorhandler(401)
def unauthorized(error):
    return render_template("errors/401.html", title="Error 401"), 401


@errors.app_errorhandler(403)
def forbidden(error):
    return render_template("errors/403.html", title="Error 403"), 403


@errors.app_errorhandler(404)
def not_found_error(error):
    return render_template("errors/404.html", title="Error 404"), 404


@errors.app_errorhandler(405)
def bad_method(error):
    return render_template("errors/405.html", title="Error 405"), 405


@errors.app_errorhandler(500)
def server_error(error):
    return render_template("errors/500.html", title="Error 500"), 500
