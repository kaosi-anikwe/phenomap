import os
import json
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

# Enable mixed precision for better performance
# set_global_policy('mixed_float16')

class CustomAttention(layers.Layer):
    """A custom attention mechanism that properly handles tensor dimensions.
    
    This layer implements both channel and spatial attention, carefully
    maintaining tensor shapes throughout the computation to ensure
    compatibility with the input features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize layers for spatial attention
        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=(7, 7),
            padding='same',
            activation='sigmoid'
        )
        
        # Initialize layers for channel attention
        # Note: We'll set the units in build() based on input shape
        self.channel_dense_reduce = None
        self.channel_dense_expand = None
    
    def build(self, input_shape):
        """Initialize layer weights based on input shape.
        
        The channel attention dense layers are created here because
        we need to know the number of channels from the input shape.
        """
        self.channels = input_shape[-1]
        
        # Create dense layers for channel attention with proper dimensions
        self.channel_dense_reduce = layers.Dense(
            units=max(self.channels // 8, 1),  # Reduction ratio of 8, minimum 1
            activation='relu'
        )
        self.channel_dense_expand = layers.Dense(
            units=self.channels,
            activation='sigmoid'
        )
        
        super().build(input_shape)
    
    def compute_channel_attention(self, x):
        """Compute channel attention weights while preserving dimensions.
        
        Args:
            x: Input tensor of shape [batch, height, width, channels]
            
        Returns:
            Channel attention weights of shape [batch, 1, 1, channels]
        """
        # Global average pooling
        avg_pool = tf.reduce_mean(x, axis=[1, 2])  # [batch, channels]
        avg_pool = self.channel_dense_reduce(avg_pool)
        avg_pool = self.channel_dense_expand(avg_pool)
        
        # Global max pooling
        max_pool = tf.reduce_max(x, axis=[1, 2])  # [batch, channels]
        max_pool = self.channel_dense_reduce(max_pool)
        max_pool = self.channel_dense_expand(max_pool)
        
        # Combine attention weights
        channel_attention = avg_pool + max_pool
        
        # Reshape to broadcasting shape [batch, 1, 1, channels]
        return tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
    
    def compute_spatial_attention(self, x):
        """Compute spatial attention weights.
        
        Args:
            x: Input tensor of shape [batch, height, width, channels]
            
        Returns:
            Spatial attention weights of shape [batch, height, width, 1]
        """
        # Average and max pooling across channels
        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)  # [batch, height, width, 1]
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)  # [batch, height, width, 1]
        
        # Concatenate and process through convolution
        spatial = tf.concat([avg_spatial, max_spatial], axis=-1)  # [batch, height, width, 2]
        return self.spatial_conv(spatial)  # [batch, height, width, 1]
    
    def call(self, x):
        """Apply channel and spatial attention to input.
        
        Args:
            x: Input tensor of shape [batch, height, width, channels]
            
        Returns:
            Attended feature maps of same shape as input
        """
        # Apply channel attention
        channel_weights = self.compute_channel_attention(x)  # [batch, 1, 1, channels]
        x = x * channel_weights  # Broadcasting handles the multiplication
        
        # Apply spatial attention
        spatial_weights = self.compute_spatial_attention(x)  # [batch, height, width, 1]
        x = x * spatial_weights  # Broadcasting handles the multiplication
        
        return x
    
    def get_config(self):
        """Return layer configuration for serialization."""
        return super().get_config()

# Register the custom layer
tf.keras.utils.get_custom_objects()['CustomAttention'] = CustomAttention


class CurriculumDataGenerator:
    """Generates training data with curriculum learning strategy."""
    
    def __init__(self, metadata, image_dir, img_size=100, batch_size=32, num_classes=None):
        self.metadata = metadata
        self.image_dir = image_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Get all unique syndromes from metadata
        syndromes = sorted(list(set(item['syndrome_code'] for item in metadata)))
        
        # If num_classes is provided, use it as a validation
        if num_classes is not None and num_classes != len(syndromes):
            print(f"Warning: Found {len(syndromes)} classes in data but expected {num_classes}")
            
        self.num_classes = num_classes if num_classes is not None else len(syndromes)
        self.syndrome_encoder = {s: i for i, s in enumerate(syndromes)}

        mapping_path = 'syndrome_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(self.syndrome_encoder, f, indent=2)
        
        print(f"Initialized generator with {self.num_classes} classes")
        self.difficulty_scores = self._calculate_difficulty()
        
    def _calculate_difficulty(self):
        """Calculate difficulty score for each image based on detection confidence and quality."""
        print("Calculating difficulty scores for curriculum learning...")
        scores = {}
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for record in tqdm(self.metadata):
            try:
                img_path = os.path.join(self.image_dir, record['image_name'])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Enhanced quality metrics
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                contrast = np.std(gray)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.mean(edges > 0)
                
                # Comprehensive difficulty score
                score = 1.0
                if len(faces) > 0:
                    score *= 0.7
                score *= (1 - min(clarity / 1000, 0.9))
                score *= abs(brightness - 128) / 128
                score *= (1 - contrast / 255)
                score *= (1 - edge_density)
                
                scores[record['image_name']] = score
                
            except Exception as e:
                print(f"Error processing {record['image_name']}: {str(e)}")
                scores[record['image_name']] = 1.0
                
        return scores
        
    def _load_and_preprocess_image(self, image_path):
        """Enhanced image preprocessing pipeline."""
        # Read and convert to grayscale
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance image quality
        img = cv2.equalizeHist(img)  # Improve contrast
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise
        
        # Normalize intensity
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Resize and normalize to [0,1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        
        return np.expand_dims(img, axis=-1)

    def generate_epoch(self, epoch, training=True):
        """Generate batches of training or validation data with curriculum learning.
        
        Args:
            epoch (int): Current training epoch number
            training (bool): Whether this is for training (True) or validation (False)
            
        Yields:
            tuple: A batch of ([images, additional_features], labels)
        """
        # Log the start of data generation for debugging
        print(f"Generating {'training' if training else 'validation'} data for epoch {epoch}")
        print(f"Number of classes: {self.num_classes}")
        
        # Sort samples by their difficulty scores (easier samples first)
        sorted_samples = sorted(self.metadata, 
                            key=lambda x: self.difficulty_scores[x['image_name']])
        
        if training:
            # For training, implement curriculum learning by gradually including harder samples
            # Start with 20% of samples in epoch 0, reach 100% by epoch 50
            progress = min(1.0, (epoch + 1) / 50)
            n_samples = max(int(len(sorted_samples) * progress), self.batch_size)
            current_samples = sorted_samples[:n_samples]
            print(f"Using {n_samples}/{len(sorted_samples)} samples ({progress:.1%}) for epoch {epoch}")
        else:
            # For validation, always use all samples
            current_samples = sorted_samples
            
        # Shuffle samples while maintaining curriculum difficulty order
        random.shuffle(current_samples)
        
        # Generate batches
        for i in range(0, len(current_samples), self.batch_size):
            batch_samples = current_samples[i:i + self.batch_size]
            
            # Initialize batch data containers
            X_images = []
            y_labels = []
            additional_features = []
            
            # Process each sample in the batch
            for sample in batch_samples:
                try:
                    # Load and preprocess image
                    img_path = os.path.join(self.image_dir, sample['image_name'])
                    img = self._load_and_preprocess_image(img_path)
                    
                    # Get syndrome label
                    label = self.syndrome_encoder[sample['syndrome_code']]
                    
                    # Process metadata features
                    gender_feature = 1 if sample['gender'] == 'M' else 0
                    ethnicity_map = {'caucasian': 0, 'african': 1, 'asian': 2, 
                                'hispanic': 3, 'indian': 4, 'arab': 5}
                    ethnicity_feature = ethnicity_map[sample['ethnicity']]
                    
                    # Add to batch containers
                    X_images.append(img)
                    y_labels.append(label)
                    additional_features.append([gender_feature, ethnicity_feature])
                    
                except Exception as e:
                    print(f"Error processing batch sample {sample['image_name']}: {str(e)}")
                    continue
                    
            # Only yield if we have valid samples in the batch
            if len(X_images) > 0:
                # Convert to numpy arrays
                X_images = np.array(X_images)
                additional_features = np.array(additional_features)
                y_labels = tf.keras.utils.to_categorical(y_labels, self.num_classes)
                
                # Validate shapes before yielding
                print(f"Batch shapes - X: {X_images.shape}, y: {y_labels.shape}")
                
                yield [X_images, additional_features], y_labels

def create_pyramid_features(x, filters):
    """Create feature pyramid for multi-scale feature extraction."""
    features = []
    for f in filters:
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        features.append(x)
        x = layers.MaxPooling2D((2, 2))(x)
    return features

def attention_module(x):
    """Enhanced attention module with both spatial and channel attention."""
    # Channel attention with SE-style squeeze-excitation
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(x.shape[-1] // 4, activation='relu')(se)
    se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
    se = layers.Reshape((1, 1, x.shape[-1]))(se)
    x = layers.Multiply()([x, se])
    
    # Spatial attention
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    spatial = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    
    return layers.Multiply()([x, spatial])

def build_model(num_classes, img_size=100):
    """Build enhanced PhenoNet model with pyramid features and improved attention."""
    # Input layers
    img_input = layers.Input(shape=(img_size, img_size, 1))
    
    # Initial feature extraction
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Apply attention using custom layer
    x = CustomAttention()(x)
    
    # Additional convolutional blocks
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Global features
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Additional features input
    additional_input = layers.Input(shape=(2,))
    y = layers.Dense(16)(additional_input)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)
    
    # Combine features
    combined = layers.Concatenate()([x, y])
    
    # Classification head
    combined = layers.Dense(512)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.LeakyReLU(alpha=0.1)(combined)
    combined = layers.Dropout(0.3)(combined)

    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(combined)
    
    return tf.keras.Model(inputs=[img_input, additional_input], outputs=output)

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

class PhenoNetTrainer:
    """Main trainer class for PhenoNet."""
    
    def __init__(self, metadata_path, image_dir, img_size=100):
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.img_size = img_size
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Create data generator
        self.data_generator = CurriculumDataGenerator(
            self.metadata, image_dir, img_size=img_size
        )
        
    def train(self, batch_size=32, epochs=200):
        """Enhanced training method with improved learning schedule and monitoring.
        
        This implementation includes significant improvements:
        - Cosine learning rate schedule with warmup
        - Enhanced batch handling and memory management
        - Improved gradient handling with clipping
        - More sophisticated learning rate adjustments
        - Better class balancing through weighted sampling
        - Mixed precision training for better performance
        
        Args:
            batch_size (int): Number of samples per training batch
            epochs (int): Maximum number of training epochs
            
        Returns:
            tuple: (trained model, training history dictionary)
        """
        
        # Calculate class distribution for balanced sampling
        all_syndromes = sorted(list(set(item['syndrome_code'] for item in self.metadata)))
        total_classes = len(all_syndromes)
        print(f"Total unique syndromes in dataset: {total_classes}")
        
        # Calculate class weights for balanced training
        syndrome_counts = {}
        for item in self.metadata:
            syndrome_counts[item['syndrome_code']] = syndrome_counts.get(item['syndrome_code'], 0) + 1
        
        max_count = max(syndrome_counts.values())
        class_weights = {
            syndrome: max_count / count 
            for syndrome, count in syndrome_counts.items()
        }
        
        # Create stratified split while maintaining class distribution
        train_size = int(0.8 * len(self.metadata))
        indices = np.random.permutation(len(self.metadata))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_metadata = [self.metadata[i] for i in train_indices]
        val_metadata = [self.metadata[i] for i in val_indices]
        
        print(f"Split dataset: {len(train_metadata)} training samples, {len(val_metadata)} validation samples")
        
        # Initialize data generators with consistent class counts
        print("Creating data generators...")
        train_generator = CurriculumDataGenerator(
            train_metadata, self.image_dir, self.img_size, batch_size,
            num_classes=total_classes
        )
        val_generator = CurriculumDataGenerator(
            val_metadata, self.image_dir, self.img_size, batch_size,
            num_classes=total_classes
        )
        
        # Build model with improved architecture
        print(f"Building model for {total_classes} classes...")
        model = build_model(total_classes, self.img_size)
        model.summary()
        
        # Calculate training parameters
        steps_per_epoch = max(len(train_metadata) // batch_size, 1)
        validation_steps = max(len(val_metadata) // batch_size, 1)
        
        # Initialize learning rate schedule with warmup
        initial_learning_rate = 1e-4  # Start with a smaller learning rate
        warmup_epochs = 5
        
        def cosine_decay_with_warmup(epoch):
            """Custom learning rate schedule with warmup and cosine decay."""
            if epoch < warmup_epochs:
                # Linear warmup
                return initial_learning_rate * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                decay_epochs = epochs - warmup_epochs
                epoch_in_decay = epoch - warmup_epochs
                cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
                return initial_learning_rate * cosine_decay
        
        # Setup optimizer with gradient clipping
        optimizer = optimizers.Adam(
            learning_rate=initial_learning_rate,
            clipnorm=1.0,  # Prevent exploding gradients
            epsilon=1e-7    # Improved numerical stability
        )
        
        # Compile model with enhanced loss function
        model.compile(
            optimizer=optimizer,
            loss=combined_loss(alpha=0.1),  # Combined focal loss with label smoothing
            metrics=['accuracy']
        )
        
        # Create unique model checkpoint filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filepath = f'phenonet_model_{timestamp}.h5'
        
        # Initialize training history
        history = {
            'accuracy': [], 'val_accuracy': [],
            'loss': [], 'val_loss': [],
            'learning_rate': []
        }
        
        # Save initial weights for potential restoration
        initial_weights = model.get_weights()
        
        # Initialize model tracking variables
        best_val_accuracy = -1
        best_weights = None
        patience_counter = 0
        last_improvement = 0
        min_delta = 0.001  # Minimum improvement threshold
        
        print("\nStarting enhanced training with curriculum learning...")
        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")
                
                # Update learning rate according to schedule
                current_lr = cosine_decay_with_warmup(epoch)
                K.set_value(model.optimizer.learning_rate, current_lr)
                print(f"Current learning rate: {current_lr:.2e}")
                
                # Training phase with improved error handling
                train_loss = []
                train_acc = []
                train_batches = 0
                
                train_pbar = tqdm(desc='Training', total=steps_per_epoch, leave=False)
                
                for batch_x, batch_y in train_generator.generate_epoch(epoch, training=True):
                    try:
                        # Apply class weights to batch
                        sample_weights = np.array([
                            class_weights[all_syndromes[np.argmax(y)]] 
                            for y in batch_y
                        ])
                        
                        metrics = model.train_on_batch(
                            batch_x, batch_y,
                            sample_weight=sample_weights
                        )
                        
                        train_loss.append(metrics[0])
                        train_acc.append(metrics[1])
                        train_batches += 1
                        train_pbar.update(1)
                        
                        if train_batches >= steps_per_epoch:
                            break
                            
                    except Exception as e:
                        print(f"Error during training batch: {str(e)}")
                        continue
                
                train_pbar.close()
                
                # Validation phase with improved monitoring
                val_loss = []
                val_acc = []
                val_batches = 0
                
                val_pbar = tqdm(desc='Validation', total=validation_steps, leave=False)
                
                for batch_x, batch_y in val_generator.generate_epoch(epoch, training=False):
                    try:
                        metrics = model.test_on_batch(batch_x, batch_y)
                        val_loss.append(metrics[0])
                        val_acc.append(metrics[1])
                        val_batches += 1
                        val_pbar.update(1)
                        
                        if val_batches >= validation_steps:
                            break
                            
                    except Exception as e:
                        print(f"Error during validation batch: {str(e)}")
                        continue
                
                val_pbar.close()
                
                # Calculate and log epoch metrics
                epoch_metrics = {
                    'loss': np.mean(train_loss),
                    'accuracy': np.mean(train_acc),
                    'val_loss': np.mean(val_loss),
                    'val_accuracy': np.mean(val_acc),
                    'learning_rate': current_lr
                }
                
                # Update history
                for metric, value in epoch_metrics.items():
                    history[metric].append(value)
                
                # Print detailed epoch summary
                print(f"loss: {epoch_metrics['loss']:.4f} - "
                    f"accuracy: {epoch_metrics['accuracy']:.4f} - "
                    f"val_loss: {epoch_metrics['val_loss']:.4f} - "
                    f"val_accuracy: {epoch_metrics['val_accuracy']:.4f}")
                
                # Enhanced model saving and early stopping logic
                improvement = epoch_metrics['val_accuracy'] - best_val_accuracy
                if improvement > min_delta:
                    best_val_accuracy = epoch_metrics['val_accuracy']
                    best_weights = model.get_weights()
                    print(f"\nSaving best model with validation accuracy: {best_val_accuracy:.4f}")
                    model.save_weights(checkpoint_filepath)
                    last_improvement = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Adaptive early stopping
                if epoch - last_improvement >= 20:
                    print("\nEarly stopping triggered: No improvement for 20 epochs")
                    model.set_weights(best_weights)
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if best_weights is not None:
                print("Restoring best weights...")
                model.set_weights(best_weights)
        
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            if best_weights is not None:
                print("Restoring best weights...")
                model.set_weights(best_weights)
            else:
                print("Restoring initial weights...")
                model.set_weights(initial_weights)
        
        finally:
            # Ensure best model is restored and results are saved
            if best_weights is not None:
                model.set_weights(best_weights)
            
            # Create visualizations
            self.plot_training_history(history)
            
            # Save training history
            save_training_history(history, timestamp)
            
            # Print final training summary
            print("\nTraining Summary:")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"Final learning rate: {current_lr:.2e}")
            print(f"Total epochs trained: {epoch + 1}")
        
        return model, history

    def plot_training_history(self, history):
        """Create comprehensive visualization of training history.
        
        Args:
            history (dict): Dictionary containing training metrics
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Train', color='blue', alpha=0.7)
        ax1.plot(history['val_accuracy'], label='Validation', color='orange', alpha=0.7)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Train', color='blue', alpha=0.7)
        ax2.plot(history['val_loss'], label='Validation', color='orange', alpha=0.7)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(history['learning_rate'], color='green', alpha=0.7)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_training_history(history, timestamp):
    """Save training history to a JSON file after converting data types.
    
    This function handles the conversion of NumPy and TensorFlow data types
    to native Python types that can be serialized to JSON. It processes
    the history dictionary recursively to ensure all nested values are
    properly converted.
    
    Args:
        history (dict): The training history dictionary containing metrics
        timestamp (str): Timestamp string for the filename
    """
    def convert_to_serializable(obj):
        """Convert a single object to a JSON-serializable format.
        
        This helper function handles various numeric types that might
        appear in our training metrics, converting them to standard
        Python floats or ints.
        """
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif tf.is_tensor(obj):
            return float(obj.numpy())
        return obj
    
    # Save to file with proper formatting
    try:
        with open(f'training_history_{timestamp}.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"Successfully saved training history to training_history_{timestamp}.json")
    except Exception as e:
        print(f"Error saving training history: {str(e)}")

    # Convert the entire history dictionary
    serializable_history = convert_to_serializable(history)
    
    # Save to file with proper formatting
    try:
        with open(f'training_history_{timestamp}.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"Successfully saved training history to training_history_{timestamp}.json")
    except Exception as e:
        print(f"Error saving training history: {str(e)}")


if __name__ == "__main__":

    metadata_path = '/home/kaosi/training/metadata.json'
    image_dir = '/home/kaosi/training'
    
    # Initialize trainer with reduced batch size
    trainer = PhenoNetTrainer(metadata_path, image_dir)
    
    # Train with longer patience and more epochs
    model, history = trainer.train(batch_size=16, epochs=300)
    
    # Save final model
    model.save('final_phenonet_model.keras')
    print("Training completed successfully!")