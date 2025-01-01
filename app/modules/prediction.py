# python imports
import time
from typing import Optional, Dict
from datetime import datetime

# installed imports
import numpy as np
from PIL import Image
from flask import Flask
import tensorflow as tf

# local imports
from .. import logger, create_app
from .secure_image_handler import SecureImageHandler
from ..models import (
    Case,
    ClassificationStatus,
    ClassificationRequest,
    Prediction,
    CasePredictionDiagnosis,
)


class GeneticDisorderClassifier:
    def __init__(self, model_path="path/to/your/model"):
        # Load model
        self.model = self.load_model(model_path)

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

    def classify(self, case: Case):
        """Perform classification for a case."""
        try:
            logger.info("Classifying image...")
            predictions = []
            for img in case.patient_images:
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

                # Process model outputs
                syndrome_predictions = self.process_model_output(
                    model_output, case["gender"], case["ethnicity"]
                )

                predictions.extend(syndrome_predictions)

            # Aggregate predictions from multiple images if needed
            logger.info(f"Got {len(predictions)} predictions")
            final_predictions = self.aggregate_predictions(predictions)

            return final_predictions

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None

    def process_model_output(self, model_output, gender, ethnicity):
        """Process raw model output into syndrome predictions."""
        try:
            # This will depend on your model's output format
            # Example implementation:
            syndrome_scores = model_output[
                0
            ]  # Assuming output is probabilities per syndrome
            predictions = []
            for idx, score in enumerate(syndrome_scores):
                if score > 0.1:  # Minimum confidence threshold
                    syndrome_info = self.get_syndrome_info(idx)
                    predictions.append(
                        {
                            "syndrome_name": syndrome_info["name"],
                            "syndrome_code": syndrome_info["code"],
                            "confidence_score": float(score),
                            "status": self.get_confidence_status(score),
                            "composite_image": syndrome_info["composite_image"],
                            "diagnosis_status": {
                                "differential": score > 0.7,
                                "clinically_diagnosed": False,
                                "molecularly_diagnosed": False,
                            },
                        }
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

    def aggregate_predictions(self, predictions):
        """Aggregate predictions from multiple images."""
        # Group by syndrome and take highest confidence score
        logger.info("Aggregating predictions")
        syndrome_dict = {}
        for pred in predictions:
            syndrome_code = pred["syndrome_code"]
            if (
                syndrome_code not in syndrome_dict
                or pred["confidence_score"]
                > syndrome_dict[syndrome_code]["confidence_score"]
            ):
                syndrome_dict[syndrome_code] = pred

        # Sort by confidence score
        final_predictions = list(syndrome_dict.values())
        final_predictions.sort(key=lambda x: x["confidence_score"], reverse=True)

        return final_predictions


class ClassificationManager:
    def __init__(self):
        self.active_requests = {}  # Store in-progress requests
        self.request_timeout = 300  # 5 minutes timeout
        # self.classifier = GeneticDisorderClassifier()

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

                now = datetime.now()
                logger.info(f"Classification began at: {now}")
                # TODO: Perform classification
                # predictions = await self.classifier.classify(case)

                # TODO: Remove simulation
                time.sleep(10)
                predictions = generate_mock_predictions()
                logger.info("Classification end")
                logger.info(f"Got {len(predictions)} predictions")
                logger.info(
                    f"Classification took {(datetime.now() - now).total_seconds()}"
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
    """Generate mock predictions for testing."""
    return [
        {
            "syndrome_name": "Prader-Willi Syndrome",
            "syndrome_code": "PWS",
            "confidence_score": 0.92,
            "status": "high",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": True,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Down Syndrome",
            "syndrome_code": "DS",
            "confidence_score": 0.88,
            "status": "high",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": True,
                "molecularly_diagnosed": True,
            },
        },
        {
            "syndrome_name": "Cornelia de Lange Syndrome",
            "syndrome_code": "CdLS",
            "confidence_score": 0.75,
            "status": "high",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Williams Syndrome",
            "syndrome_code": "WS",
            "confidence_score": 0.67,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Noonan Syndrome",
            "syndrome_code": "NS",
            "confidence_score": 0.61,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Fragile X Syndrome",
            "syndrome_code": "FXS",
            "confidence_score": 0.58,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": True,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Smith-Magenis Syndrome",
            "syndrome_code": "SMS",
            "confidence_score": 0.52,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Angelman Syndrome",
            "syndrome_code": "AS",
            "confidence_score": 0.48,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Sotos Syndrome",
            "syndrome_code": "SoS",
            "confidence_score": 0.43,
            "status": "med",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Turner Syndrome",
            "syndrome_code": "TS",
            "confidence_score": 0.39,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Rett Syndrome",
            "syndrome_code": "RTT",
            "confidence_score": 0.35,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Kabuki Syndrome",
            "syndrome_code": "KS",
            "confidence_score": 0.32,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "CHARGE Syndrome",
            "syndrome_code": "CS",
            "confidence_score": 0.28,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Treacher Collins Syndrome",
            "syndrome_code": "TCS",
            "confidence_score": 0.25,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
        {
            "syndrome_name": "Apert Syndrome",
            "syndrome_code": "AS",
            "confidence_score": 0.22,
            "status": "low",
            "composite_image": "https://phenomap.braintext.io/static/img/down.png",
            "diagnosis_status": {
                "differential": False,
                "clinically_diagnosed": False,
                "molecularly_diagnosed": False,
            },
        },
    ]
