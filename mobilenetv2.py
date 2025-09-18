import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

class OptimizedMRIClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=16):
        """
        Initialize the optimized MRI classifier
        
        Args:
            data_dir (str): Path to data directory containing 'Benign' and 'Malignant' folders
            img_size (tuple): Target image size for model input
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_weight_dict = None
        
    def setup_data_generators(self, validation_split=0.15):
        """
        Setup data generators with error handling
        """
        try:
            # Enhanced data augmentation for training set
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2],
                channel_shift_range=0.1,
                fill_mode='reflect',
                validation_split=validation_split
            )
            
            # Only rescaling for validation set
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            # Training generator
            self.train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            # Validation generator
            self.val_generator = val_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            print(f"Training samples: {self.train_generator.samples}")
            print(f"Validation samples: {self.val_generator.samples}")
            print(f"Class indices: {self.train_generator.class_indices}")
            
            # Verify we have both classes
            if len(self.train_generator.class_indices) < 2:
                raise ValueError("Binary classification requires 2 classes, but only found 1. Check your data directory structure.")
            
            # Calculate class weights for imbalanced datasets
            self.calculate_class_weights()
            
        except Exception as e:
            print(f"Error setting up data generators: {e}")
            raise
        
    def calculate_class_weights(self):
        """
        Calculate class weights with proper error handling
        """
        try:
            from sklearn.utils.class_weight import compute_class_weight
            
            # Get class counts from directory structure instead of iterating through batches
            class_counts = {}
            for class_name, class_idx in self.train_generator.class_indices.items():
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                    class_counts[class_idx] = count
            
            print(f"Class counts: {class_counts}")
            
            if len(class_counts) >= 2:
                # Create labels array based on class counts
                labels = []
                for class_idx, count in class_counts.items():
                    labels.extend([class_idx] * count)
                
                labels = np.array(labels)
                classes = np.unique(labels)
                
                self.class_weights = compute_class_weight(
                    'balanced', 
                    classes=classes, 
                    y=labels
                )
                self.class_weight_dict = dict(zip(classes.astype(float), self.class_weights))
            else:
                # Default balanced weights if calculation fails
                self.class_weight_dict = {0.0: 1.0, 1.0: 1.0}
            
            print(f"Class weights: {self.class_weight_dict}")
            
        except Exception as e:
            print(f"Error calculating class weights, using default balanced weights: {e}")
            self.class_weight_dict = {0.0: 1.0, 1.0: 1.0}
        
    def create_model(self, learning_rate=0.001):
        """
        Create model with comprehensive error handling
        """
        try:
            # Load pre-trained MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            # Freeze base model layers initially
            base_model.trainable = False
            
            # Enhanced classification head with regularization
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(256, activation='relu', 
                     kernel_regularizer=l2(0.001),
                     bias_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            
            x = Dense(128, activation='relu',
                     kernel_regularizer=l2(0.001),
                     bias_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            predictions = Dense(1, activation='sigmoid')(x)
            
            # Create the model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile with optimized Adam optimizer
            optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=True
            )
            
            # Use string names for metrics to avoid version issues
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'AUC']
            )
            
            print("Model created successfully!")
            print(f"Total parameters: {self.model.count_params():,}")
            
            # Calculate trainable parameters safely
            try:
                trainable_params = sum([np.prod(v.shape.as_list()) for v in self.model.trainable_variables])
                print(f"Trainable parameters: {trainable_params:,}")
            except:
                try:
                    trainable_params = sum([np.prod(v.shape) for v in self.model.trainable_variables])
                    print(f"Trainable parameters: {trainable_params:,}")
                except:
                    print("Trainable parameters: Could not calculate")
                    
            return self.model
            
        except Exception as e:
            print(f"Error creating model: {e}")
            raise
        
    def train_model_simple(self, epochs=50):
        """
        Training with comprehensive error handling
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")
                
            if self.class_weight_dict is None:
                raise ValueError("Class weights not calculated. Call setup_data_generators() first.")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',  # Use val_loss instead of val_auc for more reliability
                    patience=10,
                    restore_best_weights=True,
                    mode='min',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_mri_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
            ]
            
            print("Starting training...")
            
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                class_weight=self.class_weight_dict,
                verbose=1
            )
            
            print("Training completed!")
            return self.history
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        
    def plot_training_history(self):
        """
        Plot training history with robust error handling
        """
        try:
            if self.history is None:
                print("No training history available. Train the model first.")
                return
            
            # Get available metrics
            available_metrics = list(self.history.history.keys())
            print("Available metrics:", available_metrics)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot accuracy
            if 'accuracy' in available_metrics:
                axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
                if 'val_accuracy' in available_metrics:
                    axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                axes[0, 0].set_title('Model Accuracy')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            else:
                axes[0, 0].text(0.5, 0.5, 'Accuracy not available', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Model Accuracy (Not Available)')
            
            # Plot loss
            if 'loss' in available_metrics:
                axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
                if 'val_loss' in available_metrics:
                    axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
                axes[0, 1].set_title('Model Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            else:
                axes[0, 1].text(0.5, 0.5, 'Loss not available', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Model Loss (Not Available)')
            
            # Plot AUC - check for various naming conventions
            auc_key = None
            val_auc_key = None
            
            possible_auc_names = ['auc', 'AUC', 'auc_1', 'auc_2', 'binary_accuracy']
            possible_val_auc_names = ['val_auc', 'val_AUC', 'val_auc_1', 'val_auc_2', 'val_binary_accuracy']
            
            for name in possible_auc_names:
                if name in available_metrics:
                    auc_key = name
                    break
            
            for name in possible_val_auc_names:
                if name in available_metrics:
                    val_auc_key = name
                    break
            
            if auc_key:
                axes[1, 0].plot(self.history.history[auc_key], label=f'Training AUC ({auc_key})', linewidth=2)
                if val_auc_key:
                    axes[1, 0].plot(self.history.history[val_auc_key], label=f'Validation AUC ({val_auc_key})', linewidth=2)
                axes[1, 0].set_title('Model AUC')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'AUC not available\nAvailable metrics:\n' + '\n'.join(available_metrics), 
                               ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=8)
                axes[1, 0].set_title('Model AUC (Not Available)')
            
            # Plot precision and recall
            precision_key = None
            val_precision_key = None
            recall_key = None
            val_recall_key = None
            
            possible_precision_names = ['precision', 'precision_1', 'precision_2']
            possible_val_precision_names = ['val_precision', 'val_precision_1', 'val_precision_2']
            possible_recall_names = ['recall', 'recall_1', 'recall_2']
            possible_val_recall_names = ['val_recall', 'val_recall_1', 'val_recall_2']
            
            for name in possible_precision_names:
                if name in available_metrics:
                    precision_key = name
                    break
            
            for name in possible_val_precision_names:
                if name in available_metrics:
                    val_precision_key = name
                    break
                    
            for name in possible_recall_names:
                if name in available_metrics:
                    recall_key = name
                    break
            
            for name in possible_val_recall_names:
                if name in available_metrics:
                    val_recall_key = name
                    break
            
            if precision_key or recall_key:
                if precision_key:
                    axes[1, 1].plot(self.history.history[precision_key], label=f'Training Precision ({precision_key})', linewidth=2)
                    if val_precision_key:
                        axes[1, 1].plot(self.history.history[val_precision_key], label=f'Val Precision ({val_precision_key})', linewidth=2)
                
                if recall_key:
                    axes[1, 1].plot(self.history.history[recall_key], label=f'Training Recall ({recall_key})', linewidth=2)
                    if val_recall_key:
                        axes[1, 1].plot(self.history.history[val_recall_key], label=f'Val Recall ({val_recall_key})', linewidth=2)
                
                axes[1, 1].set_title('Precision & Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Precision/Recall not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Precision & Recall (Not Available)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
        
    def evaluate_model(self):
        """
        Evaluate model with comprehensive error handling
        """
        try:
            if self.model is None:
                print("No model available. Create and train the model first.")
                return None
                
            # Reset generator
            self.val_generator.reset()
            
            # Get predictions with proper batch handling
            print("Making predictions...")
            
            # Calculate number of steps needed
            steps = np.ceil(self.val_generator.samples / self.val_generator.batch_size)
            
            # Get predictions
            val_predictions = self.model.predict(self.val_generator, steps=steps, verbose=1)
            val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()
            
            # Get true labels properly
            self.val_generator.reset()
            true_labels = []
            
            for i in range(int(steps)):
                try:
                    batch_x, batch_y = next(self.val_generator)
                    true_labels.extend(batch_y)
                except StopIteration:
                    break
            
            true_labels = np.array(true_labels)
            
            # Ensure same length
            min_len = min(len(val_predictions_binary), len(true_labels))
            val_predictions_binary = val_predictions_binary[:min_len]
            true_labels = true_labels[:min_len]
            val_predictions = val_predictions[:min_len]
            
            print(f"Evaluation samples: {min_len}")
            
            # Classification report
            print("\nClassification Report:")
            print("=" * 50)
            print(classification_report(true_labels, val_predictions_binary, 
                                      target_names=['Benign', 'Malignant']))
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, val_predictions_binary)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant'], 
                       yticklabels=['Benign', 'Malignant'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            
            # ROC Curve
            if len(np.unique(true_labels)) > 1:  # Only plot ROC if both classes present
                fpr, tpr, _ = roc_curve(true_labels, val_predictions.flatten())
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.show()
                
                return roc_auc
            else:
                print("Cannot compute ROC curve - only one class present in validation set")
                return None
                
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None
        
    def save_model(self, filepath='mri_classifier_model.h5'):
        """
        Save model with error handling
        """
        try:
            if self.model is None:
                print("No model to save. Train the model first.")
                return False
                
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

# Example usage with comprehensive error handling
if __name__ == "__main__":
    try:
        # Update this path to point to your Raw_Annotated_Samples folder
        data_directory = "/content/ovarian_data/Raw_Annotated_Samples"
        
        # Check if directory exists
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        # Initialize classifier
        classifier = OptimizedMRIClassifier(data_directory, img_size=(224, 224), batch_size=16)
        
        # Setup data generators
        classifier.setup_data_generators(validation_split=0.15)
        
        # Create model
        model = classifier.create_model(learning_rate=0.001)
        
        # Display model summary
        model.summary()
        
        # Train model
        history = classifier.train_model_simple(epochs=20)
        
        # Plot training history
        classifier.plot_training_history()
        
        # Evaluate model
        auc_score = classifier.evaluate_model()
        
        # Save model
        success = classifier.save_model('ovarian_mri_classifier.h5')
        
        if auc_score is not None:
            print(f"\nFinal AUC Score: {auc_score:.4f}")
        else:
            print("\nAUC Score could not be calculated")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please check your data directory structure and try again.")