import os
import logging
from flask import Flask
from flask_wtf import CSRFProtect
from flask_login import LoginManager
from config import Config
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

load_dotenv()

log_dir = os.getenv("LOG_DIR")

# create folder if not exists
os.makedirs(log_dir, exist_ok=True)

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
csrf = CSRFProtect()
migrate = Migrate()

# configure logger
logging.basicConfig(
    filename=os.path.join(log_dir, "run.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("phenomap")


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


    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(api)

    csrf.exempt(api)

    return app
