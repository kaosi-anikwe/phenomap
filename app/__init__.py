import os
import logging
from flask import Flask
from config import Config
from dotenv import load_dotenv
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from logging.handlers import RotatingFileHandler

load_dotenv()

log_dir = os.getenv("LOG_DIR")
upload_dir = os.getenv("UPLOADS")

# create folder if not exists
os.makedirs(log_dir, exist_ok=True)
os.makedirs(upload_dir, exist_ok=True)

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
csrf = CSRFProtect()
migrate = Migrate()

# Configure logger
log_filename = "run.log"
log_max_size = 1 * 1024 * 1024  # 1 MB

# Create a logger
logger = logging.getLogger("phenomap")
logger.setLevel(logging.INFO)

# Create a file handler with log rotation
handler = RotatingFileHandler(
    os.path.join(log_dir, log_filename), maxBytes=log_max_size, backupCount=5
)

# Create a formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


def create_app(config=Config):
    app = Flask(__name__)

    app.config.from_object(config)
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    migrate.init_app(app, db)

    from app.main.routes import main
    from app.auth.routes import auth
    from app.api.routes import api
    from app.images.routes import images
    from app.errors.handlers import errors

    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(errors)
    app.register_blueprint(api, url_prefix="/api/cases")
    app.register_blueprint(images, url_prefix="/api/images")

    csrf.exempt(api)
    csrf.exempt(images)

    return app
