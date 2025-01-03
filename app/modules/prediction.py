# python imports
import time
import random
from datetime import datetime
from typing import Optional, Dict

# installed imports
import numpy as np
from PIL import Image
import tensorflow as tf

# local imports
from .. import logger, create_app
from .secure_image_handler import SecureImageHandler
from ..models import (
    Case,
    Syndrome,
    Prediction,
    ClassificationStatus,
    ClassificationRequest,
    CasePredictionDiagnosis,
)


class GeneticDisorderClassifier:
    def __init__(self, model_path="path/to/your/model"):
        # Load model
        # self.model = self.load_model(model_path)
        pass

    def load_model(self, model_path):
        """Load the trained model."""
        try:
            # Replace with your actual model loading code
            logger.info(f"Loading model from path: {model_path}")
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        try:
            logger.info("Preprocessing image...")
            image = Image.open(image)
            # TODO: Add your image preprocessing steps here
            # For example:
            image = image.resize((224, 224))  # Resize to model input size
            image_array = np.array(image)
            # TODO: Add any normalization or other preprocessing steps
            return image_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def classify(self, case: Case, fake=True):
        """Perform classification for a case."""
        try:
            logger.info("Classifying image...")
            predictions = []
            for idx, img in enumerate(case.patient_images):
                logger.info(f"Predicting image {idx}")
                if not fake:
                    # Preprocess image
                    handler = SecureImageHandler(img.encryption_key)
                    processed_image = self.preprocess_image(
                        handler.retrieve_image(img.id, case.user_id)
                    )

                    if processed_image is None:
                        continue

                    # Get model predictions
                    model_output = self.model.predict(
                        np.expand_dims(processed_image, axis=0)
                    )
                else:
                    model_output = generate_mock_predictions()

                predictions.extend(model_output)

                logger.info(f"Got {len(predictions)} predictions")

            logger.info(f"Done predicting {len(case.patient_images)} images")
            # Process model outputs
            syndrome_predictions = self.process_model_output(
                self.remove_duplicates(predictions)
            )
            logger.info(f"Total: {len(syndrome_predictions)} predictions")

            return syndrome_predictions

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None

    def process_model_output(self, model_output):
        """Process raw model output into syndrome predictions."""
        try:
            predictions = []
            for output in model_output:
                if output["score"] > 0.1:  # Minimum confidence threshold
                    syndrome = Syndrome.query.filter(
                        Syndrome.code == output["code"]
                    ).first()
                    if syndrome:
                        predictions.append(
                            {
                                "syndrome_name": syndrome.title,
                                "syndrome_code": syndrome.code,
                                "confidence_score": float(output["score"]),
                                "status": self.get_confidence_status(output["score"]),
                                "composite_image": syndrome.composite_image
                                or "img/down.png",
                                "diagnosis_status": {
                                    "differential": output["score"] > 0.7,
                                    "clinically_diagnosed": False,
                                    "molecularly_diagnosed": False,
                                },
                            }
                        )
                    else:
                        logger.info(
                            f"Skipping missing syndrome prediction with code: {output['code']}"
                        )

            return predictions
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            return []

    def get_confidence_status(self, score):
        """Convert confidence score to status level."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "med"
        return "low"

    def remove_duplicates(self, data):
        # Create a dictionary to store highest score for each code
        highest_scores = {}
        # Iterate through the list
        for item in data:
            code = item["code"]
            score = item["score"]
            # If code not in dictionary or new score is higher, update it
            if code not in highest_scores or score > highest_scores[code]["score"]:
                highest_scores[code] = item
        # Convert dictionary values back to list
        return list(highest_scores.values())


class ClassificationManager:
    def __init__(self):
        self.active_requests = {}  # Store in-progress requests
        self.request_timeout = 300  # 5 minutes timeout
        self.classifier = GeneticDisorderClassifier()

    def create_request(self, case_id: str, force: bool = False) -> dict:
        """Create a new classification request."""
        try:
            logger.info(f"Attempting to create classification request")
            # Check for existing active request
            existing_request = self._get_latest_request(case_id)
            if not force and existing_request:
                if existing_request["status"] in ["pending", "processing"]:
                    raise ValueError("Classification already in progress")
                elif existing_request["status"] == "completed":
                    raise ValueError(
                        "Classification already exists. Use force=True to reclassify"
                    )
            # Create database record
            logger.info(f"Adding new classification request")
            request = ClassificationRequest()
            request.case_id = case_id
            request.prerequisites_met = True
            request.insert()
            # Track the request
            self.active_requests[case_id] = {
                "id": request.id,
                "started_at": datetime.utcnow(),
                "status": ClassificationStatus.PENDING.value,
            }
            return request.json()
        except Exception as e:
            raise RuntimeError(f"Failed to create classification request: {str(e)}")

    def process_request(self, request_id: int) -> None:
        """Process a classification request."""
        app = create_app()
        with app.app_context():
            try:
                logger.info(f"Processing request with ID: {request_id}")
                # Update status to processing
                self._update_request_status(request_id, ClassificationStatus.PROCESSING)

                # Get case data
                case = ClassificationRequest.query.get(request_id).request_case
                # Track time
                now = datetime.now()
                logger.info(f"Classification began at: {now}")
                predictions = self.classifier.classify(case)
                logger.info(
                    f"Classification end: {(datetime.now() - now).total_seconds()}"
                )

                # Store predictions
                self._store_predictions(request_id, predictions)

                # Update request status
                self._update_request_status(request_id, ClassificationStatus.COMPLETED)

                # Clean up active request tracking
                self._cleanup_request(case.id)

            except Exception as e:
                self._update_request_status(
                    request_id, ClassificationStatus.FAILED, str(e)
                )
                raise

    def get_request_status(self, request_id: int) -> dict:
        """Get the current status of a classification request."""
        logger.info(f"Getting status for ClassificationRequest #{request_id}")
        return ClassificationRequest.query.get(request_id).json()

    def _store_predictions(self, request_id: int, predictions: list) -> None:
        """Store predictions in database."""
        logger.info(f"Storing predictions in Database")
        try:
            case = ClassificationRequest.query.get(request_id).request_case
            existing_predictions = case.predictions
            for idx, pred in enumerate(predictions):
                logger.info(f"Prediction -> Syndrome Name: {pred['syndrome_name']}")
                logger.info(f"Prediction -> Syndrome Code: {pred['syndrome_code']}")
                exists = any(
                    [
                        pr.id
                        for pr in existing_predictions
                        if pr.syndrome_name == pred["syndrome_name"]
                        and pr.syndrome_code == pred["syndrome_code"]
                    ]
                )
                if not exists:
                    # Prediction not already associated with case
                    logger.info(f"Adding syndrome")
                    new_pred = Prediction()
                    new_pred.case_id = case.id
                    new_pred.syndrome_name = pred["syndrome_name"]
                    new_pred.syndrome_code = pred["syndrome_code"]
                    new_pred.confidence_score = pred["confidence_score"]
                    new_pred.status = pred["status"]
                    new_pred.composite_image = pred["composite_image"]
                    new_pred.insert()
                    predictions[idx]["id"] = new_pred.id

                    # Store diagnosis status
                    diagnosis = CasePredictionDiagnosis()
                    diagnosis.prediction_id = new_pred.id
                    diagnosis.differential = pred["diagnosis_status"]["differential"]
                    diagnosis.clinically_diagnosed = pred["diagnosis_status"][
                        "clinically_diagnosed"
                    ]
                    diagnosis.molecularly_diagnosed = pred["diagnosis_status"][
                        "molecularly_diagnosed"
                    ]
                    diagnosis.insert()
                else:
                    logger.info("Syndrome already associated with case")
        except Exception as e:
            logger.error(f"Failed to store predictions: {str(e)}")
            raise RuntimeError(f"Failed to store predictions: {str(e)}")

    def _update_request_status(
        self, request_id: int, status: ClassificationStatus, error_message: str = None
    ) -> None:
        """Update the status of a request."""
        logger.info(f"Updating request #{request_id} with status: {status}")
        request = ClassificationRequest.query.get(request_id)
        request.status = status.value
        request.error = error_message
        request.completed_at = (
            datetime.utcnow()
            if status.value in ["completed", "failed"]
            else request.created_at
        )
        request.update()

    def _get_latest_request(self, case_id: str) -> Optional[Dict]:
        """Get the latest classification request for a case."""
        logger.info(f"Getting latest status for Case #{case_id}")
        request = ClassificationRequest.query.filter(
            ClassificationRequest.case_id == case_id
        ).first()
        return request.json() if request else None

    def _cleanup_request(self, case_id: str) -> None:
        """Remove request from active tracking."""
        logger.info(f"Cleaning up Case #{case_id} from active requests")
        if case_id in self.active_requests:
            del self.active_requests[case_id]


def generate_mock_predictions():
    """Generate random predictions for testing."""
    time.sleep(10)
    syndromes = Syndrome.query.all()
    random.shuffle(syndromes)
    syndromes = syndromes[:20]
    scores = [round(random.uniform(0.0, 0.9), 5) for _ in syndromes]
    return [
        {"code": synd.code, "score": scores[idx]} for idx, synd in enumerate(syndromes)
    ]
