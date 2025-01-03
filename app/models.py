# python imports
import time
import json
import uuid
from enum import Enum
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


class ClassificationStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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
    prediction_requests = db.relationship(
        "ClassificationRequest", backref="request_case", cascade="delete,all"
    )

    def __init__(self, user_id):
        self.uid = uuid.uuid4().hex[:5]
        self.user_id = user_id

    def __repr__(self):
        return f"<Case: {self.name}>"

    def date_of_birth(self):
        return self.dob.strftime("%d %b %Y") if self.dob else None

    def image_id(self):
        return next((img.id for img in self.patient_images if img.is_default), None)

    def check_classification_prerequisites(self):
        result = Case.query.filter(Case.id == self.id).one()
        return {
            "can_classify": all(
                [result.gender, result.ethnicity, result.patient_images]
            ),
            "missing_prerequisites": {
                "gender": not result.gender,
                "ethnicity": not result.ethnicity,
                "images": not result.patient_images,
            },
        }

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
    syndrome_name = db.Column(db.String(256), nullable=False)
    syndrome_code = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(10), nullable=False)
    composite_image = db.Column(db.Text)
    is_removed = db.Column(db.Boolean, default=False)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"))
    syndrome_id = db.Column(db.Integer, db.ForeignKey("syndrome.id"))
    diagnosis = db.relationship(
        "CasePredictionDiagnosis", backref="case", cascade="delete", uselist=False
    )

    def __repr__(self):
        return f"<Prediction {self.id} - Case {self.case_id}>"

    def json(self):
        return {
            "syndrome_name": self.syndrome_name,
            "syndrome_code": self.syndrome_code,
            "composite_image": self.composite_image,
            "case_photo_id": self.case.image_id(),
            "status": self.status,
            "confidence_score": self.confidence_score,
            "diagnosis_status": {
                "differential_diagnosed": self.diagnosis.differential,
                "molecularly_diagnosed": self.diagnosis.molecularly_diagnosed,
                "clinically_diagnosed": self.diagnosis.clinically_diagnosed,
            },
        }

    def syndrome(self):
        return Syndrome.query.get(self.syndrome_id)


class CasePredictionDiagnosis(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "prediction_diagnosis"

    id = db.Column(db.Integer, primary_key=True)
    differential = db.Column(db.Boolean, default=False)
    clinically_diagnosed = db.Column(db.Boolean, default=False)
    molecularly_diagnosed = db.Column(db.Boolean, default=False)
    prediction_id = db.Column(db.ForeignKey("prediction.id"), unique=True)


class ClassificationRequest(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "classification_requests"

    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(10), default=ClassificationStatus.PENDING.value)
    completed_at = db.Column(db.DateTime)
    success = db.Column(db.Boolean)
    error = db.Column(db.Text)
    prerequisites_met = db.Column(db.Boolean)
    case_id = db.Column(db.ForeignKey("case.id"))

    def json(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


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


class Syndrome(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "syndrome"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(256), nullable=False)
    code = db.Column(db.String(10), nullable=False)
    composite_image = db.Column(db.String(256), default="img/down.png")
    synonyms = db.Column(db.Text)
    omim = db.Column(db.String(20))
    genes = db.Column(db.String(1024))
    location = db.Column(db.String(50))
    images = db.Column(db.Text)
    inheritance_modes = db.Column(db.String(1024))
    abstract = db.Column(db.Text)
    features = db.Column(db.Text)
    resources = db.Column(db.Text)

    def __init__(
        self,
        title,
        code,
        synonyms,
        omim,
        genes,
        location,
        images,
        inheritance_modes,
        abstract,
        features,
        resources,
    ):
        self.title = title
        self.code = code
        self.synonyms = synonyms
        self.omim = omim
        self.genes = genes
        self.location = location
        self.images = images
        self.inheritance_modes = inheritance_modes
        self.abstract = abstract
        self.features = features
        self.resources = resources

    def get_synonyms(self):
        return str(self.synonyms).split(",")

    def get_genes(self):
        return str(self.genes).split(",")

    def get_inheritance_modes(self):
        return str(self.inheritance_modes).split(",")

    def get_features(self):
        return str(self.features).split(",")

    def get_resources(self):
        return json.loads(self.resources) if self.resources else {}

    def get_images(self):
        image_records = json.loads(self.images) if self.images else []
        images = [img["path"] for img in image_records if img.get("path")]
        return images or ["img/down.png"]

    def json(self):
        return {
            "title": self.title,
            "synonyms": self.get_synonyms(),
            "composite_image": self.composite_image or "/img/down.png",
            "omim": self.omim,
            "genes": self.get_genes(),
            "location": self.location,
            "images": self.get_images(),
            "inheritance_modes": self.get_inheritance_modes(),
            "abstract": self.abstract,
            "features": self.get_features(),
            "resources": self.get_resources(),
        }
