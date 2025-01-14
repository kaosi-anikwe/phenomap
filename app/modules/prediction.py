# python imports
import os
import time
import json
import random
import traceback
from datetime import datetime
from typing import Optional, Dict

# installed imports
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

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


class PhenoNetPredictor:
    """A class for making predictions with a trained PhenoNet model.

    This class handles all the preprocessing and prediction logic needed to use
    PhenoNet in a production environment. It maintains consistent preprocessing
    with the training pipeline while providing a simple interface for predictions.
    """

    def __init__(
        self,
        model_path="model/phenonet_model_20250104_155140.h5",
        syndrome_mapping_path="model/syndrome_map.json",
    ):
        """Initialize the predictor with a trained model and syndrome mappings.

        Args:
            model_path: Path to the saved .keras or .h5 model file
            syndrome_mapping_path: Path to the JSON file containing syndrome mappings
        """
        # Load the trained model
        # self.model = tf.keras.models.load_model(
        #     model_path,
        #     custom_objects={'loss_function': self._dummy_loss},  # Handle custom loss,
        #     safe_mode=False,
        # )
        self.model = load_checkpoint_model(model_path, 602)
        print("Model loaded successfully and ready for predictions")

        # Load syndrome mappings
        with open(syndrome_mapping_path, "r") as f:
            self.syndrome_mapping = json.load(f)

        # Reverse the mapping for predictions
        self.index_to_syndrome = {v: k for k, v in self.syndrome_mapping.items()}

        # Store model parameters
        self.img_size = 100  # Same as training

    def _dummy_loss(self, y_true, y_pred):
        """Dummy loss function for model loading - not used for predictions."""
        return 0

    def preprocess_image(self, image_data):
        """Preprocess an image from bytes data for syndrome prediction.

        This method handles image preprocessing for both file uploads and memory streams,
        applying the same preprocessing steps used during model training to ensure
        consistent predictions.

        Args:
            image_data: BytesIO object containing the image data
                This could come from a web upload, API request, or other stream

        Returns:
            numpy.ndarray: Preprocessed image array ready for model input,
                with shape (1, img_size, img_size, 1) and values normalized to [0,1]

        Raises:
            ValueError: If the image data cannot be properly decoded or processed
        """
        try:
            # Convert BytesIO data to numpy array
            nparr = np.frombuffer(image_data.getvalue(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode image data")

            # Convert to grayscale - crucial for syndrome analysis
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing steps to enhance facial features
            img = cv2.equalizeHist(img)  # Enhance contrast for better feature detection
            img = cv2.GaussianBlur(
                img, (3, 3), 0
            )  # Reduce noise while preserving edges

            # Resize to model's expected input size
            img = cv2.resize(img, (self.img_size, self.img_size))

            # Normalize pixel values to [0,1] range
            img = img.astype("float32") / 255.0

            # Add batch and channel dimensions for model input
            img = np.expand_dims(img, axis=(0, -1))

            return img

        except Exception as e:
            # Provide detailed error information for debugging
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def classify(self, case: Case, top_k=15):
        """Make a prediction for a single image.

        Args:
            image_path: Path to the image file
            gender: 'M' or 'F'
            ethnicity: One of ['caucasian', 'african', 'asian', 'hispanic', 'indian']
            top_k: Number of top predictions to return

        Returns:
            List of (syndrome, confidence) tuples for top k predictions
        """
        try:
            logger.info("Classifying case...")
            # Perform classification for each image
            model_predictions = []
            for idx, img in enumerate(case.patient_images):
                logger.info(f"Predicting image {idx}")

                # Preprocess image
                handler = SecureImageHandler(img.encryption_key)
                img_data = handler.retrieve_image(img.id, case.user_id)
                img = self.preprocess_image(img_data)

                # Prepare metadata features
                gender_feature = 1 if case.gender[:1].upper() == "M" else 0
                ethnicity_map = {
                    "caucasian": 0,
                    "african": 1,
                    "asian": 2,
                    "hispanic": 3,
                    "indian": 4,
                    "arab": 5,
                }
                ethnicity_feature = ethnicity_map[case.ethnicity.lower()]

                # Create metadata input
                metadata = np.array([[gender_feature, ethnicity_feature]])

                # Make prediction
                predictions = self.model.predict([img, metadata], verbose=0)

                # Get top k predictions
                top_indices = np.argsort(predictions[0])[-top_k:][::-1]

                logger.info(f"PREDICTIONS: {top_indices}")

                # Format results
                results = []
                for idx in top_indices:
                    try:
                        syndrome = self.index_to_syndrome[idx]
                    except KeyError:
                        logger.info(f"Skipping unknown syndrome: {idx}")
                        continue
                    confidence = float(predictions[0][idx])
                    logger.info(f"OUTPUT: {(syndrome, confidence)}")
                    results.append({"code": syndrome, "score": confidence * 10})

                logger.info(f"Got {len(results)} predictions")
                model_predictions.extend(results)
            logger.info(f"Done predicting {len(case.patient_images)} images")
            # Process model outputs
            final_predictions = self.process_model_output(
                self.remove_duplicates(model_predictions)
            )
            logger.info(f"Total: {len(final_predictions)} predictions")
            return final_predictions
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


def focal_loss(gamma=2.0, alpha=0.25):
    """Custom focal loss function that properly handles tensor shapes.

    This implementation uses reduction operations correctly and ensures
    tensor compatibility throughout the calculation.

    Args:
        gamma (float): Focusing parameter for harder examples
        alpha (float): Weight factor for class imbalance

    Returns:
        function: Loss function that can be used by Keras
    """

    def loss_function(y_true, y_pred):
        # Add small epsilon to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Convert logits to probabilities if needed
        if not isinstance(y_pred, tf.Tensor):
            y_pred = tf.convert_to_tensor(y_pred)
        if not isinstance(y_true, tf.Tensor):
            y_true = tf.convert_to_tensor(y_true)

        # Ensure both tensors are float32
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Calculate focal weight
        focal_weight = tf.pow(1 - y_pred, gamma)

        # Combine all factors
        focal_loss = alpha * focal_weight * ce

        # Reduce mean across all dimensions
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss_function


def combined_loss(alpha=0.1):
    """Enhanced loss function combining focal loss with label smoothing."""
    focal = focal_loss(gamma=2.0)

    def loss_function(y_true, y_pred):
        # Label smoothing
        eps = 0.1
        y_true_smooth = y_true * (1 - eps) + eps / y_true.shape[-1]

        # Combine losses
        focal_term = focal(y_true, y_pred)
        ce_term = tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
        return focal_term + alpha * ce_term

    return loss_function


def load_checkpoint_model(checkpoint_path, num_classes=602):
    """Load a model from a checkpoint file (.h5 format).

    This function handles loading a model from weights by:
    1. Recreating the original model architecture
    2. Loading the saved weights into this architecture
    3. Setting up the model for inference

    Args:
        checkpoint_path: Path to the .h5 checkpoint file
        num_classes: Number of classes the model was trained on

    Returns:
        A compiled Keras model ready for making predictions
    """
    # First, recreate the original model architecture
    model = build_model(num_classes=num_classes)

    # Load the weights from the checkpoint
    try:
        # Try loading just the weights
        model.load_weights(checkpoint_path)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        raise

    # Compile the model for inference

    model.compile(
        optimizer="adam",  # The optimizer doesn't matter for inference
        loss=combined_loss(alpha=0.1),
        metrics=["accuracy"],
    )

    return model


def create_pyramid_features(x, filters):
    """Create feature pyramid for multi-scale feature extraction."""
    features = []
    for f in filters:
        x = layers.Conv2D(f, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        features.append(x)
        x = layers.MaxPooling2D((2, 2))(x)
    return features


def attention_module(x):
    """Enhanced attention module with both spatial and channel attention."""
    # Channel attention with SE-style squeeze-excitation
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(x.shape[-1] // 4, activation="relu")(se)
    se = layers.Dense(x.shape[-1], activation="sigmoid")(se)
    se = layers.Reshape((1, 1, x.shape[-1]))(se)
    x = layers.Multiply()([x, se])

    # Spatial attention
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    spatial = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(concat)

    return layers.Multiply()([x, spatial])


def build_model(num_classes, img_size=100):
    """Build enhanced PhenoNet model with pyramid features and improved attention."""
    # Input layers
    img_input = layers.Input(shape=(img_size, img_size, 1))

    # Initial feature extraction
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Feature pyramid
    pyramid_features = create_pyramid_features(x, [64, 128, 256])

    # Global context
    context = layers.GlobalAveragePooling2D()(pyramid_features[-1])
    context = layers.Dense(256)(context)
    context = layers.BatchNormalization()(context)
    context = layers.LeakyReLU(alpha=0.1)(context)

    # Attention on final pyramid feature
    attended = attention_module(pyramid_features[-1])

    # Global features
    x = layers.GlobalAveragePooling2D()(attended)
    x = layers.Concatenate()([x, context])

    # Additional features input
    additional_input = layers.Input(shape=(2,))
    y = layers.Dense(32)(additional_input)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)
    y = layers.Dropout(0.2)(y)

    # Combine features
    combined = layers.Concatenate()([x, y])

    # Classification head
    combined = layers.Dense(512)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.LeakyReLU(alpha=0.1)(combined)
    combined = layers.Dropout(0.3)(combined)

    combined = layers.Dense(256)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.LeakyReLU(alpha=0.1)(combined)
    combined = layers.Dropout(0.3)(combined)

    output = layers.Dense(num_classes, activation="softmax")(combined)

    return models.Model(inputs=[img_input, additional_input], outputs=output)
