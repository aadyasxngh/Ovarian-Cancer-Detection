import os, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------------------------
# Custom Logit Pruning Layer
# -------------------------------------------------
class LogitPruningLayer(layers.Layer):
    """Custom layer that zeros out the lower 30% of logits after sorting in descending order"""
    
    def __init__(self, prune_ratio=0.3, **kwargs):
        super(LogitPruningLayer, self).__init__(**kwargs)
        self.prune_ratio = prune_ratio
        
    def call(self, inputs):
        # Get batch size and feature dimension
        batch_size = tf.shape(inputs)[0]
        feature_dim = tf.shape(inputs)[1]
        
        # Calculate number of features to zero out
        num_to_zero = tf.cast(tf.cast(feature_dim, tf.float32) * self.prune_ratio, tf.int32)
        
        # Sort indices in descending order of values
        sorted_indices = tf.argsort(inputs, axis=1, direction='DESCENDING')
        
        # Create range for batch indexing
        batch_indices = tf.range(batch_size)
        batch_indices = tf.expand_dims(batch_indices, 1)
        batch_indices = tf.tile(batch_indices, [1, num_to_zero])
        
        # Get indices of lowest values (last num_to_zero indices after sorting)
        lowest_indices = sorted_indices[:, -num_to_zero:]
        
        # Combine batch and feature indices for scatter_nd
        indices_to_zero = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(lowest_indices, [-1])
        ], axis=1)
        
        # Create mask with zeros for pruned positions
        mask = tf.ones_like(inputs)
        updates = tf.zeros(tf.shape(indices_to_zero)[0])
        mask = tf.tensor_scatter_nd_update(mask, indices_to_zero, updates)
        
        # Apply mask to inputs
        return inputs * mask
    
    def get_config(self):
        config = super(LogitPruningLayer, self).get_config()
        config.update({'prune_ratio': self.prune_ratio})
        return config

# -------------------------------------------------
# Custom Focal Loss (for binary classification)
# -------------------------------------------------
def focal_loss(alpha=0.75, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = alpha * tf.pow(1. - pt, gamma) * bce
        return tf.reduce_mean(fl)
    return loss_fn


# -------------------------------------------------
# Enhanced GoogLeNet with Logit Pruning
# -------------------------------------------------
class EnhancedGoogLeNetClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, prune_ratio=0.3):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.prune_ratio = prune_ratio
        self.model = None
        self.class_names = None
        self.history = None
        self.history_fine = None
        self.class_weights = None

    # Split dataset into train/val/test
    def prepare_splits(self, output_dir="split_dataset", test_split=0.15, val_split=0.15):
        os.makedirs(output_dir, exist_ok=True)
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(images)

            n_total = len(images)
            n_test = int(n_total * test_split)
            n_val = int(n_total * val_split)

            splits = {
                "test": images[:n_test],
                "val": images[n_test:n_test+n_val],
                "train": images[n_test+n_val:]
            }

            for split, files in splits.items():
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                for f in files:
                    shutil.copyfile(os.path.join(class_path, f), os.path.join(split_class_dir, f))
        return output_dir

    # Data generators
    def setup_data_generators(self, split_dir):
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest"
        )
        datagen_val_test = ImageDataGenerator(rescale=1./255)

        self.train_generator = datagen_train.flow_from_directory(
            os.path.join(split_dir, "train"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary"
        )
        self.val_generator = datagen_val_test.flow_from_directory(
            os.path.join(split_dir, "val"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary"
        )
        self.test_generator = datagen_val_test.flow_from_directory(
            os.path.join(split_dir, "test"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False
        )
        self.class_names = list(self.train_generator.class_indices.keys())

        # Compute class weights for imbalance
        labels = self.train_generator.classes
        self.class_weights = dict(enumerate(compute_class_weight("balanced", classes=np.unique(labels), y=labels)))

    # Build model with logit pruning layers
    def create_model(self, learning_rate=1e-3, dropout1=0.5, dropout2=0.7):
        base_model = applications.InceptionV3(weights="imagenet", include_top=False,
                                              input_shape=(*self.img_size, 3))
        base_model.trainable = False  # freeze initially

        x = layers.GlobalAveragePooling2D()(base_model.output)
        
        # First dense layer with pruning
        x = layers.Dense(512, activation="relu", name="dense_512")(x)
        x = layers.BatchNormalization(name="bn_512")(x)
        x = LogitPruningLayer(prune_ratio=self.prune_ratio, name="prune_512")(x)  # Prune after first dense
        x = layers.Dropout(dropout1, name="dropout_512")(x)
        
        # Second dense layer with pruning
        x = layers.Dense(256, activation="relu", name="dense_256")(x)
        x = layers.BatchNormalization(name="bn_256")(x)
        x = LogitPruningLayer(prune_ratio=self.prune_ratio, name="prune_256")(x)  # Prune after second dense
        x = layers.Dropout(dropout2, name="dropout_256")(x)
        
        # Final prediction layer (no pruning on output)
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

        self.model = models.Model(inputs=base_model.input, outputs=predictions)

        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-4, nesterov=True)
        self.model.compile(optimizer=opt,
                           loss=focal_loss(alpha=0.75, gamma=2.0),
                           metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        
        print(f"\n🔧 Model created with logit pruning ratio: {self.prune_ratio}")
        print(f"📊 Pruning layers will zero out lowest {self.prune_ratio*100:.0f}% of activations")
        
        return self.model

    # Train model with early stopping patience 8
    def train_model(self, epochs=100):
        # Updated callbacks with early stopping patience 8
        callbacks = [
            EarlyStopping(
                monitor="val_auc", 
                patience=8,  # Changed from 25 to 8
                restore_best_weights=True, 
                min_delta=0.001,
                verbose=1,
                mode="max"
            ),
            ModelCheckpoint(
                "best_model_with_30_pruning.h5", 
                save_best_only=True, 
                monitor="val_auc", 
                mode="max",
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_auc", 
                factor=0.5, 
                patience=5,  # Reduced from 10 to 5 for faster adaptation
                min_lr=1e-7, 
                verbose=1, 
                mode="max"
            )
        ]
        
        print("🚀 Phase 1: Feature Extraction (frozen base) with 30% Logit Pruning")
        print(f"⏱️  Early stopping patience: 8 epochs")
        
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=30,  # Phase 1
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )

        print("\n🔓 Phase 2: Fine-tuning last 50 layers with 30% Logit Pruning...")
        
        # Unfreeze last 50 layers
        for layer in self.model.layers[-50:]:
            if not isinstance(layer, layers.BatchNormalization):  
                layer.trainable = True

        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9, decay=1e-5),
            loss=focal_loss(alpha=0.75, gamma=2.0),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        fine_tune_epochs = epochs - 30
        print(f"⚡ Fine-tuning for {fine_tune_epochs} epochs with early stopping...")
        
        self.history_fine = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=fine_tune_epochs,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history, self.history_fine

    # Analyze pruning effect
    def analyze_pruning_effect(self, num_samples=10):
        """Analyze the sparsity introduced by logit pruning layers"""
        print("\n🔍 Analyzing 30% Logit Pruning Effect")
        print("=" * 50)
        
        # Get a batch of test data
        test_batch = next(iter(self.test_generator))
        x_batch, _ = test_batch
        x_sample = x_batch[:num_samples]
        
        # Create intermediate models to get activations
        prune_layer_names = [layer.name for layer in self.model.layers if 'prune' in layer.name]
        
        for prune_layer_name in prune_layer_names:
            # Get activations before pruning (from the layer just before)
            before_layer_name = None
            for i, layer in enumerate(self.model.layers):
                if layer.name == prune_layer_name and i > 0:
                    before_layer_name = self.model.layers[i-1].name
                    break
            
            if before_layer_name:
                # Model to get activations before pruning
                before_model = models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(before_layer_name).output
                )
                
                # Model to get activations after pruning
                after_model = models.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(prune_layer_name).output
                )
                
                # Get activations
                before_activations = before_model.predict(x_sample, verbose=0)
                after_activations = after_model.predict(x_sample, verbose=0)
                
                # Calculate sparsity
                zeros_before = np.sum(before_activations == 0)
                zeros_after = np.sum(after_activations == 0)
                total_elements = before_activations.size
                
                sparsity_before = zeros_before / total_elements * 100
                sparsity_after = zeros_after / total_elements * 100
                additional_sparsity = sparsity_after - sparsity_before
                
                print(f"\n📍 {prune_layer_name}:")
                print(f"   Sparsity before pruning: {sparsity_before:.1f}%")
                print(f"   Sparsity after pruning:  {sparsity_after:.1f}%")
                print(f"   Additional sparsity:     {additional_sparsity:.1f}%")
                print(f"   Pruned features per sample: {int(before_activations.shape[1] * self.prune_ratio)}/{before_activations.shape[1]}")

    # Plot training history with enhanced visualization
    def plot_training_history(self):
        # Combine histories from both phases
        acc = self.history.history['accuracy'] + self.history_fine.history['accuracy']
        val_acc = self.history.history['val_accuracy'] + self.history_fine.history['val_accuracy']
        auc = self.history.history['auc'] + self.history_fine.history['auc']
        val_auc = self.history.history['val_auc'] + self.history_fine.history['val_auc']
        loss = self.history.history['loss'] + self.history_fine.history['loss']
        val_loss = self.history.history['val_loss'] + self.history_fine.history['val_loss']

        epochs_range = range(len(acc))
        phase1_end = len(self.history.history['accuracy'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy plot
        axes[0,0].plot(epochs_range, acc, 'b-', label='Train Acc', alpha=0.8)
        axes[0,0].plot(epochs_range, val_acc, 'r-', label='Val Acc', alpha=0.8)
        axes[0,0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning starts')
        axes[0,0].set_title('Training Accuracy (with 30% Logit Pruning)')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[0,1].plot(epochs_range, auc, 'b-', label='Train AUC', alpha=0.8)
        axes[0,1].plot(epochs_range, val_auc, 'r-', label='Val AUC', alpha=0.8)
        axes[0,1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning starts')
        axes[0,1].set_title('Training AUC (Early Stop Patience=8)')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('AUC')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1,0].plot(epochs_range, loss, 'b-', label='Train Loss', alpha=0.8)
        axes[1,0].plot(epochs_range, val_loss, 'r-', label='Val Loss', alpha=0.8)
        axes[1,0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning starts')
        axes[1,0].set_title('Training Loss (Focal Loss)')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available in history)
        if 'lr' in self.history.history:
            lr = self.history.history['lr'] + self.history_fine.history['lr']
            axes[1,1].plot(epochs_range, lr, 'g-', label='Learning Rate', alpha=0.8)
            axes[1,1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5)
            axes[1,1].set_title('Learning Rate Schedule')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            # Show training summary instead
            axes[1,1].text(0.1, 0.8, f"Training Summary:", transform=axes[1,1].transAxes, fontsize=12, fontweight='bold')
            axes[1,1].text(0.1, 0.7, f"• Total epochs: {len(acc)}", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].text(0.1, 0.6, f"• Early stopping patience: 8", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].text(0.1, 0.5, f"• Logit pruning ratio: {self.prune_ratio*100:.0f}%", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].text(0.1, 0.4, f"• Best val AUC: {max(val_auc):.4f}", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].text(0.1, 0.3, f"• Best val accuracy: {max(val_acc):.4f}", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')

        plt.tight_layout()
        plt.suptitle('Enhanced GoogLeNet Training with 30% Logit Pruning & Early Stopping', 
                     fontsize=16, y=1.02)
        plt.show()

    # Enhanced evaluation with pruning analysis
    def evaluate_model(self, model_path="best_model_with_30_pruning.h5"):
        print("\n🎯 Evaluating Best Model with 30% Logit Pruning")
        print("=" * 50)
        
        # Load the best model with custom objects
        custom_objects = {
            "loss_fn": focal_loss(),
            "LogitPruningLayer": LogitPruningLayer
        }
        
        best_model = models.load_model(model_path, custom_objects=custom_objects)
        
        # Get predictions
        print("🔄 Generating predictions...")
        preds = best_model.predict(self.test_generator, verbose=1)
        y_pred = (preds > 0.5).astype(int).ravel()
        y_true = self.test_generator.classes

        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap="Blues")
        plt.title('Confusion Matrix (Model with 30% Logit Pruning)')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # ROC Curve
        auc_score = roc_auc_score(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, preds)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Model with 30% Logit Pruning)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Analyze pruning effect
        self.analyze_pruning_effect()
        
        return {
            'auc_score': auc_score,
            'accuracy': np.mean(y_pred == y_true),
            'predictions': preds,
            'true_labels': y_true
        }


# -------------------------------------------------
# Main script
# -------------------------------------------------
if __name__ == "__main__":
    print("🧠 Enhanced GoogLeNet with 30% Logit Pruning & Early Stopping")
    print("=" * 60)
    
    # Initialize classifier with 30% logit pruning
    classifier = EnhancedGoogLeNetClassifier(
        data_dir="/content/ovarian_data/Raw_Annotated_Samples",
        prune_ratio=0.3  # Zero out lowest 30% of logits
    )
    
    print("Preparing dataset splits...")
    split_dir = classifier.prepare_splits()
    
    print("Setting up data generators...")
    classifier.setup_data_generators(split_dir)
    
    print("Creating model with 30% logit pruning layers...")
    classifier.create_model()
    
    print("\nModel Summary:")
    classifier.model.summary()
    
    print("\nStarting training with early stopping (patience=8)...")
    classifier.train_model(epochs=100)
    
    print("\nPlotting training history...")
    classifier.plot_training_history()
    
    print("\nEvaluating final model...")
    results = classifier.evaluate_model()
    
    print(f"\nTraining Complete!")
    print(f"Final AUC Score: {results['auc_score']:.4f}")
    print(f"Final Accuracy: {results['accuracy']:.4f}")