import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

class OptimizedResNet50MRIClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=16):
        """
        Initialize the optimized ResNet50 MRI classifier
        
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
        
    def setup_data_generators(self, validation_split=0.2):
        """
        Setup data generators with optimized augmentation for ResNet50
        """
        try:
            # Enhanced data augmentation optimized for ResNet50
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.85, 1.15],
                channel_shift_range=0.08,
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
            
            # Get class counts from directory structure
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
        
    def create_model(self, learning_rate=0.0001):
        """
        Create ResNet50 model with optimized architecture and hyperparameters
        """
        try:
            # Load pre-trained ResNet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            # Fine-tuning strategy: Freeze early layers, unfreeze later layers
            # ResNet50 has 175 layers, we'll unfreeze the last 30 layers for fine-tuning
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            for layer in base_model.layers[-30:]:
                layer.trainable = True
            
            # Optimized classification head for medical imaging
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # First dense layer
            x = Dense(512, activation='relu', 
                     kernel_regularizer=l2(0.01),
                     bias_regularizer=l2(0.01),
                     name='dense_1')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # Second dense layer
            x = Dense(256, activation='relu',
                     kernel_regularizer=l2(0.01),
                     bias_regularizer=l2(0.01),
                     name='dense_2')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            
            # Third dense layer
            x = Dense(128, activation='relu',
                     kernel_regularizer=l2(0.01),
                     bias_regularizer=l2(0.01),
                     name='dense_3')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Output layer
            predictions = Dense(1, activation='sigmoid', name='predictions')(x)
            
            # Create the model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile with optimized Adam optimizer for ResNet50
            optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                amsgrad=True
            )
            
            # Use string names for metrics to avoid version issues
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'AUC']
            )
            
            print("ResNet50 model created successfully!")
            print(f"Total parameters: {self.model.count_params():,}")
            
            # Calculate trainable parameters
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
        
    def train_model(self, epochs=20):
        """
        Training with optimized callbacks and learning rate scheduling
        """
        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")
                
            if self.class_weight_dict is None:
                raise ValueError("Class weights not calculated. Call setup_data_generators() first.")
            
            # Optimized callbacks for ResNet50
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=8,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1,
                    min_delta=0.001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=4,
                    min_lr=1e-8,
                    verbose=1,
                    cooldown=2
                ),
                ModelCheckpoint(
                    'best_resnet50_mri_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1,
                    save_weights_only=False
                )
            ]
            
            print("Starting training with ResNet50...")
            print(f"Training for {epochs} epochs...")
            
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
        Plot training history with comprehensive metrics
        """
        try:
            if self.history is None:
                print("No training history available. Train the model first.")
                return
            
            # Get available metrics
            available_metrics = list(self.history.history.keys())
            print("Available metrics:", available_metrics)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot accuracy
            if 'accuracy' in available_metrics:
                axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
                if 'val_accuracy' in available_metrics:
                    axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
                axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim([0, 1])
            
            # Plot loss
            if 'loss' in available_metrics:
                axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='red')
                if 'val_loss' in available_metrics:
                    axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='purple')
                axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot AUC
            auc_key = None
            val_auc_key = None
            
            possible_auc_names = ['auc', 'AUC', 'auc_1', 'auc_2']
            possible_val_auc_names = ['val_auc', 'val_AUC', 'val_auc_1', 'val_auc_2']
            
            for name in possible_auc_names:
                if name in available_metrics:
                    auc_key = name
                    break
            
            for name in possible_val_auc_names:
                if name in available_metrics:
                    val_auc_key = name
                    break
            
            if auc_key:
                axes[1, 0].plot(self.history.history[auc_key], label=f'Training AUC', linewidth=2, color='green')
                if val_auc_key:
                    axes[1, 0].plot(self.history.history[val_auc_key], label=f'Validation AUC', linewidth=2, color='brown')
                axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim([0, 1])
            
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
                    axes[1, 1].plot(self.history.history[precision_key], label='Training Precision', linewidth=2, color='cyan')
                    if val_precision_key:
                        axes[1, 1].plot(self.history.history[val_precision_key], label='Val Precision', linewidth=2, color='magenta')
                
                if recall_key:
                    axes[1, 1].plot(self.history.history[recall_key], label='Training Recall', linewidth=2, color='yellow', linestyle='--')
                    if val_recall_key:
                        axes[1, 1].plot(self.history.history[val_recall_key], label='Val Recall', linewidth=2, color='black', linestyle='--')
                
                axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim([0, 1])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
        
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        try:
            if self.model is None:
                print("No model available. Create and train the model first.")
                return None
                
            # Reset generator
            self.val_generator.reset()
            
            # Get predictions
            print("Making predictions on validation set...")
            
            # Calculate number of steps needed
            steps = np.ceil(self.val_generator.samples / self.val_generator.batch_size)
            
            # Get predictions
            val_predictions = self.model.predict(self.val_generator, steps=steps, verbose=1)
            val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()
            
            # Get true labels
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
            print("=" * 60)
            report = classification_report(true_labels, val_predictions_binary, 
                                         target_names=['Benign', 'Malignant'],
                                         digits=4)
            print(report)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, val_predictions_binary)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant'], 
                       yticklabels=['Benign', 'Malignant'],
                       cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix - ResNet50 MRI Classifier', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.show()
            
            # Calculate additional metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, val_predictions_binary)
            precision = precision_score(true_labels, val_predictions_binary)
            recall = recall_score(true_labels, val_predictions_binary)
            f1 = f1_score(true_labels, val_predictions_binary)
            
            print(f"\nDetailed Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # ROC Curve
            if len(np.unique(true_labels)) > 1:
                fpr, tpr, thresholds = roc_curve(true_labels, val_predictions.flatten())
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=3, 
                        label=f'ROC Curve (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve - ResNet50 MRI Classifier', fontsize=16, fontweight='bold')
                plt.legend(loc="lower right", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.show()
                
                print(f"AUC Score: {roc_auc:.4f}")
                return roc_auc
            else:
                print("Cannot compute ROC curve - only one class present in validation set")
                return None
                
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None
        
    def save_model(self, filepath='resnet50_ovarian_mri_classifier.h5'):
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

# Main execution with comprehensive error handling
if __name__ == "__main__":
    try:
        # Update this path to your data directory
        data_directory = "/content/ovarian_data/Raw_Annotated_Samples"
        
        # Check if directory exists
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        
        print("Initializing ResNet50 MRI Classifier for Ovarian Cancer Detection")
        print("=" * 70)
        
        # Initialize classifier with optimized batch size for ResNet50
        classifier = OptimizedResNet50MRIClassifier(
            data_directory, 
            img_size=(224, 224), 
            batch_size=16
        )
        
        # Setup data generators with 20% validation split
        print("\nSetting up data generators...")
        classifier.setup_data_generators(validation_split=0.2)
        
        # Create ResNet50 model with optimized hyperparameters
        print("\nCreating ResNet50 model...")
        model = classifier.create_model(learning_rate=0.0001)
        
        # Display model summary
        print("\nModel Architecture Summary:")
        model.summary()
        
        # Train model for 20 epochs
        print("\nStarting training for 20 epochs...")
        history = classifier.train_model(epochs=20)
        
        # Plot training history
        print("\nPlotting training history...")
        classifier.plot_training_history()
        
        # Evaluate model
        print("\nEvaluating model performance...")
        auc_score = classifier.evaluate_model()
        
        # Save model
        print("\nSaving trained model...")
        success = classifier.save_model('resnet50_ovarian_mri_classifier.h5')
        
        # Final results
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        if auc_score is not None:
            print(f"Final AUC Score: {auc_score:.4f}")
        
        if success:
            print("Model saved successfully!")
        
        print("\nModel is ready for inference on new MRI scans.")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please check your data directory structure and try again.")
        print("\nExpected structure:")
        print("ovarian_data/")
        print("└── Raw_Annotated_Samples/")
        print("    ├── Benign/")
        print("    │   ├── image1.jpg")
        print("    │   ├── image2.jpg")
        print("    │   └── ...")
        print("    └── Malignant/")
        print("        ├── image1.jpg")
        print("        ├── image2.jpg")
        print("        └── ...")