# python imports
import time
import uuid
from datetime import datetime

# installed imports
import jwt
from flask import current_app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# local imports
from app import db, login_manager


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


class TimestampMixin(object):
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    def format_date(self):
        return (
            self.created_at.strftime("%d %b %Y"),
            self.updated_at.strftime("%d %b %Y")
            if self.updated_at
            else self.created_at.strftime("%d %b %Y"),
            None,
        )

    def format_time(self):
        try:
            self.datetime = self.datetime.strftime("%d %b, %Y %I:%M")
        except:
            pass


# db helper functions
class DatabaseHelperMixin(object):
    def update(self):
        db.session.commit()

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


class User(db.Model, TimestampMixin, UserMixin, DatabaseHelperMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(200), unique=True, nullable=False)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    organization = db.Column(db.String(200))
    account_type = db.Column(db.String(20), default="regular")
    email_verified = db.Column(db.Boolean, default=False)
    password_hash = db.Column(db.String(2000), nullable=False)
    cases = db.relationship("Case", backref="user", cascade="delete,all")
    patient_images = db.relationship(
        "PatientImage", backref="case", cascade="delete,all"
    )

    def __init__(self, firstname, lastname, email, organization, password=None) -> None:
        self.firstname = firstname
        self.lastname = lastname
        self.email = email
        self.organization = organization
        self.password_hash = self.get_password_hash(str(password)) if password else None
        self.uid = uuid.uuid4().hex

    def __repr__(self):
        return f"<User: {self.display_name()}>"

    # generate user password i.e. hashing
    def get_password_hash(self, password):
        return generate_password_hash(password)

    # check user password is correct
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # for reseting a user password
    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {"reset_password": self.id, "exp": time() + expires_in},
            current_app.config["SECRET_KEY"],
            algorithm="HS256",
        )

    # return concatenated name
    def display_name(self):
        return f"{self.firstname} {self.lastname}"

    # verify token generated for resetting password
    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(
                token, current_app.config["SECRET_KEY"], algorithms=["HS256"]
            )["reset_password"]
        except:
            return None
        return User.query.get(id)


class Case(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "case"

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(200), unique=True, nullable=False)
    name = db.Column(db.String(200), default="")
    vist_date = db.Column(db.Date)
    dob = db.Column(db.Date)
    gender = db.Column(db.String(10))
    ethnicity = db.Column(db.String(20))
    status = db.Column(db.String(10), default="Pending")
    height = db.Column(db.Integer, default=0)
    weight = db.Column(db.Integer, default=0)
    head_circ = db.Column(db.Integer, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    predictions = db.relationship("Prediction", backref="case", cascade="delete,all")
    patient_images = db.relationship(
        "PatientImage", backref="image_case", cascade="delete,all"
    )
    notes = db.relationship("CaseNote", backref="note_case", cascade="delete,all")

    def __init__(self, user_id):
        self.uid = uuid.uuid4().hex[:5]
        self.user_id = user_id

    def __repr__(self):
        return f"<Case: {self.name}>"

    def date_of_birth(self):
        return self.dob.strftime("%d %b %Y") if self.dob else None

    def parse(self):
        default_image = PatientImage.query.filter(
            PatientImage.case_id == self.id, PatientImage.is_default == True
        ).first()
        if not default_image and self.patient_images:
            self.patient_images[0].is_default = True
            self.patient_images[0].update()
        return {
            "id": self.id,
            "uid": self.uid,
            "name": self.name if self.name else "Untitled",
            "image_count": len(self.patient_images),
            "created": self.format_date()[0],
            "modified": self.format_date()[1],
            "status": self.status,
            "image": PatientImage.query.filter(
                PatientImage.case_id == self.id, PatientImage.is_default == True
            )
            .first()
            .id
            if self.patient_images
            else "",
        }


class CaseNote(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "case_note"

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    case_id = db.Column(db.ForeignKey("case.id"))

    def __init__(self, content, case_id):
        self.content = content
        self.case_id = case_id


class Prediction(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"))

    def __init__(self, name, case_id):
        self.name = name
        self.case_id = case_id

    def __repr__(self):
        return f"<Prediction: {self.name} - Case {self.case_id}>"


class PatientImage(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "patient_image"

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), default="Frontal")
    date_taken = db.Column(db.Date)
    description = db.Column(db.String(512))
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    filename = db.Column(db.String(256), nullable=False)
    encryption_key = db.Column(db.String(256), nullable=False)
    content_hash = db.Column(db.String(256), nullable=False)
    is_deleted = db.Column(db.Boolean, default=False)
    is_default = db.Column(db.Boolean, default=False)
    deletion_date = db.Column(db.DateTime)

    def __init__(self, case_id, filename, encryption_key, content_hash, user_id):
        self.case_id = case_id
        self.user_id = user_id
        self.filename = filename
        self.encryption_key = encryption_key
        self.content_hash = content_hash
