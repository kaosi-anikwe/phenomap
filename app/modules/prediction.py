# python imports
import os
import time
import json
import random
import traceback
from datetime import datetime
from typing import Optional, Dict, List

# installed imports
import cv2
import numpy as np
import mediapipe as mp
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


# Disable GPU usage
tf.config.set_visible_devices([], 'GPU')

class PhenoNetPredictor:
    """Handles prediction using trained models"""

    def __init__(
        self, models_dir: str = "models", metadata_path: str = "models/metadata.json"
    ):
        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Create reverse mapping from index to syndrome code
        self.idx_to_code = {
            int(v): k for k, v in self.metadata["class_mapping"].items()
        }

        # Load models
        self.models = {}
        for region in ["full_face", "eyes", "nose", "mouth"]:
            model_path = os.path.join(models_dir, f"{region}_final_model.keras")
            if os.path.exists(model_path):
                self.models[region] = tf.keras.models.load_model(model_path)

        # Initialize face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

        # Define facial regions
        self.regions = {
            "full_face": list(range(0, 468)),
            "eyes": [
                33,
                133,
                157,
                158,
                159,
                160,
                161,
                173,
                246,
                362,
                385,
                386,
                387,
                388,
                466,
            ],
            "nose": [1, 2, 98, 327, 6, 5, 4, 19, 94, 19],
            "mouth": [
                0,
                267,
                269,
                270,
                409,
                291,
                375,
                321,
                405,
                314,
                17,
                84,
                181,
                91,
                146,
            ],
        }

    def preprocess_image(self, image_data) -> Optional[Dict[str, np.ndarray]]:
        """Preprocess image for prediction"""
        try:
            # Read image
            nparr = np.frombuffer(image_data.getvalue(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"Could not read image: {image_data}")

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                raise ValueError(f"No face detected in: {image_data}")

            # Get landmarks
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)

            # Extract regions
            h, w = image.shape[:2]
            regions_dict = {}

            for region_name, landmark_indices in self.regions.items():
                # Get region boundaries
                region_landmarks = landmarks[landmark_indices]
                x_min = max(0, int(np.min(region_landmarks[:, 0]) * w))
                y_min = max(0, int(np.min(region_landmarks[:, 1]) * h))
                x_max = min(w, int(np.max(region_landmarks[:, 0]) * w))
                y_max = min(h, int(np.max(region_landmarks[:, 1]) * h))

                # Add padding
                padding = 0.1
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - int(width * padding))
                x_max = min(w, x_max + int(width * padding))
                y_min = max(0, y_min - int(height * padding))
                y_max = min(h, y_max + int(height * padding))

                # Crop and preprocess
                crop = image[y_min:y_max, x_min:x_max]
                crop = cv2.resize(crop, (100, 100))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                regions_dict[region_name] = crop

            # Normalize pixel values
            for region in regions_dict:
                regions_dict[region] = regions_dict[region].astype("float32") / 255.0
                regions_dict[region] = regions_dict[region][..., np.newaxis]

            return regions_dict

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    def predict(self, image_data, top_k: int = 5, normalize: float = 0.7) -> Optional[List]:
        """Predict syndrome from image"""
        # Preprocess image
        regions = self.preprocess_image(image_data)
        if regions is None:
            return None

        # Get predictions from each model
        predictions = []
        for region_name, model in self.models.items():
            region_data = regions[region_name]
            pred = model.predict(region_data[np.newaxis, ...], verbose=0)
            predictions.append(pred[0])

        # Average predictions
        avg_pred = np.mean(predictions, axis=0)

        # Get top-k predictions
        top_indices = np.argsort(avg_pred)[-top_k:][::-1]

        results = [
            {"code": self.idx_to_code[idx], "score": min(float(avg_pred[idx]) + float(avg_pred[idx]) * normalize, 1)}
            for idx in top_indices
        ]

        return results

    def process_batch(
        self,
        image_paths: List[str],
        top_k: int = 5,
        confidence_threshold: float = 0.0,
        batch_size: int = 32,
    ) -> List[Optional[Dict]]:
        """Process a batch of images"""
        all_results = []

        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            logger.info(
                f"\nProcessing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}"
            )

            # Preprocess batch
            batch_regions = {}
            valid_indices = []
            valid_paths = []

            for j, path in enumerate(batch_paths):
                regions = self.preprocess_image(path)
                if regions is not None:
                    for region_name in self.models.keys():
                        if region_name not in batch_regions:
                            batch_regions[region_name] = []
                        batch_regions[region_name].append(regions[region_name])
                    valid_indices.append(j)
                    valid_paths.append(path)

            # Skip if no valid images in batch
            if not valid_indices:
                all_results.extend([None] * len(batch_paths))
                continue

            # Convert to numpy arrays
            for region_name in batch_regions:
                batch_regions[region_name] = np.stack(batch_regions[region_name])

            # Get predictions from each model
            batch_predictions = []
            for region_name, model in self.models.items():
                pred = model.predict(batch_regions[region_name], verbose=0)
                batch_predictions.append(pred)

            # Average predictions
            avg_preds = np.mean(batch_predictions, axis=0)

            # Process each prediction
            batch_results = []
            for idx, path in zip(valid_indices, valid_paths):
                pred = avg_preds[len(batch_results)]

                # Get top-k predictions above threshold
                top_indices = np.argsort(pred)[::-1]
                filtered_preds = []

                for pred_idx in top_indices:
                    confidence = float(pred[pred_idx])
                    if confidence >= confidence_threshold:
                        filtered_preds.append(
                            {
                                "syndrome": self.idx_to_code[pred_idx],
                                "confidence": confidence,
                            }
                        )
                        if len(filtered_preds) == top_k:
                            break

                result = {
                    "image_path": path,
                    "predictions": filtered_preds,
                    "max_confidence": float(pred[top_indices[0]]),
                }
                batch_results.append(result)

            # Fill in None for failed images
            full_batch_results = [None] * len(batch_paths)
            for idx, result in zip(valid_indices, batch_results):
                full_batch_results[idx] = result

            all_results.extend(full_batch_results)

        return all_results

    def classify(self, case: Case, top_k=15):
        try:
            logger.info("Classifying case...")
            # Perform classification for each image
            model_predictions = []
            for idx, img in enumerate(case.patient_images):
                logger.info(f"Predicting image {idx}")

                # Preprocess image
                handler = SecureImageHandler(img.encryption_key)
                image_data = handler.retrieve_image(img.id, case.user_id)
                results = self.predict(image_data, top_k, 5)
                logger.info(f"{len(results)} PREDICTIONS: {results}")
                model_predictions.extend(results)
            logger.info(f"Done predicting {len(case.patient_images)} images")
            # Process model outputs
            predictions = self.process_model_output(
                self.remove_duplicates(model_predictions)
            )
            logger.info(f"Total: {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Classification error: {e}")
            logger.error(traceback.format_exc())
            return []

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
        self.classifier = PhenoNetPredictor()

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
