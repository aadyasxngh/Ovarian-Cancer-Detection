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
import cv2
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

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
# XAI Visualization Methods
# -------------------------------------------------
class XAIVisualizer:
    """Class containing all XAI visualization methods"""
    
    def __init__(self, model, img_size=(224, 224)):
        self.model = model
        self.img_size = img_size
        
    def preprocess_image(self, image_path):
        """Load and preprocess image for model input"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array, img
    
    def compute_saliency_map(self, image_path, class_idx=0):
        """
        Compute saliency map using vanilla gradients
        """
        img_array, original_img = self.preprocess_image(image_path)
        img_tensor = tf.convert_to_tensor(img_array)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            # For binary classification, use the prediction directly
            target_class_output = predictions[0][0]
        
        # Compute gradients
        gradients = tape.gradient(target_class_output, img_tensor)
        
        # Convert to numpy and take absolute values
        gradients = gradients.numpy()[0]
        saliency = np.max(np.abs(gradients), axis=-1)
        
        # Normalize to 0-1
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency, original_img, predictions[0][0].numpy()
    
    def compute_grad_cam(self, image_path, layer_name=None):
        """
        Compute Grad-CAM visualization
        """
        img_array, original_img = self.preprocess_image(image_path)
        
        # If no layer specified, use the last convolutional layer
        if layer_name is None:
            # Find the last convolutional layer in the base model
            conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name.lower()]
            if conv_layers:
                layer_name = conv_layers[-1].name
            else:
                # Fallback to a known layer in InceptionV3
                layer_name = 'mixed10'
        
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # For binary classification
            loss = predictions[0][0]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute the guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by "how important this channel is"
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (self.img_size[1], self.img_size[0]))
        
        return heatmap, original_img, predictions[0][0].numpy()
    
    def compute_occlusion_map(self, image_path, patch_size=32, stride=16):
        """
        Compute occlusion sensitivity map
        """
        img_array, original_img = self.preprocess_image(image_path)
        
        # Get baseline prediction
        baseline_pred = self.model.predict(img_array, verbose=0)[0][0]
        
        # Initialize occlusion map
        occlusion_map = np.zeros((self.img_size[0], self.img_size[1]))
        
        # Slide the patch across the image
        for i in range(0, self.img_size[0] - patch_size + 1, stride):
            for j in range(0, self.img_size[1] - patch_size + 1, stride):
                # Create occluded image
                occluded_img = img_array.copy()
                occluded_img[0, i:i+patch_size, j:j+patch_size, :] = 0.5  # Gray out the patch
                
                # Get prediction for occluded image
                occluded_pred = self.model.predict(occluded_img, verbose=0)[0][0]
                
                # Calculate importance (difference from baseline)
                importance = baseline_pred - occluded_pred
                
                # Assign importance to all pixels in the patch
                occlusion_map[i:i+patch_size, j:j+patch_size] = importance
        
        # Normalize occlusion map
        if occlusion_map.max() > occlusion_map.min():
            occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())
        
        return occlusion_map, original_img, baseline_pred
    
    def visualize_xai_results(self, image_path, save_path=None):
        """
        Create comprehensive XAI visualization showing all three methods
        """
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        
        # Compute all XAI methods
        saliency, original_img, pred_saliency = self.compute_saliency_map(image_path)
        grad_cam, _, pred_gradcam = self.compute_grad_cam(image_path)
        occlusion, _, pred_occlusion = self.compute_occlusion_map(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title(f'Original Image\nPrediction: {pred_saliency:.3f}', fontsize=12)
        axes[0, 0].axis('off')
        
        # Saliency Map
        axes[0, 1].imshow(original_img)
        axes[0, 1].imshow(saliency, alpha=0.6, cmap='hot')
        axes[0, 1].set_title('Saliency Map\n(Vanilla Gradients)', fontsize=12)
        axes[0, 1].axis('off')
        
        # Grad-CAM
        axes[0, 2].imshow(original_img)
        axes[0, 2].imshow(grad_cam, alpha=0.6, cmap='jet')
        axes[0, 2].set_title('Grad-CAM\n(Class Activation)', fontsize=12)
        axes[0, 2].axis('off')
        
        # Saliency Map alone
        im1 = axes[1, 0].imshow(saliency, cmap='hot')
        axes[1, 0].set_title('Saliency Map\n(Isolated)', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Grad-CAM alone
        im2 = axes[1, 1].imshow(grad_cam, cmap='jet')
        axes[1, 1].set_title('Grad-CAM\n(Isolated)', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # Occlusion Map
        im3 = axes[1, 2].imshow(occlusion, cmap='RdYlBu_r')
        axes[1, 2].set_title('Occlusion Sensitivity\n(Patch-based)', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        plt.suptitle(f'XAI Analysis: {os.path.basename(image_path)}\n'
                    f'Model Prediction: {pred_saliency:.3f} (30% Logit Pruning)', 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 XAI visualization saved: {save_path}")
        
        plt.show()
        
        return {
            'saliency': saliency,
            'grad_cam': grad_cam,
            'occlusion': occlusion,
            'prediction': pred_saliency
        }


# -------------------------------------------------
# Enhanced GoogLeNet with Logit Pruning and XAI
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
        self.xai_visualizer = None

    # Split dataset into train/val/test
    def prepare_splits(self, output_dir="split_dataset", test_split=0.15, val_split=0.15):
        print(f"📁 Checking dataset directory: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")
        
        # List all items in the dataset directory
        items = os.listdir(self.data_dir)
        print(f"📋 Found {len(items)} items in dataset directory: {items}")
        
        # Find valid class directories
        valid_classes = []
        total_images = 0
        
        for class_name in items:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"⏭️  Skipping {class_name} (not a directory)")
                continue
            
            # Look for image files with common extensions
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(class_path, f))]
            
            print(f"📂 Class '{class_name}': {len(images)} images")
            
            if len(images) == 0:
                print(f"⚠️  Warning: No images found in {class_name}")
                continue
            
            valid_classes.append(class_name)
            total_images += len(images)
        
        if len(valid_classes) == 0:
            raise ValueError(f"No valid image classes found in {self.data_dir}. "
                           f"Please check your dataset structure.")
        
        if total_images == 0:
            raise ValueError(f"No images found in any class directories. "
                           f"Supported formats: {image_extensions}")
        
        print(f"✅ Found {len(valid_classes)} valid classes with {total_images} total images")
        
        # Create splits directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each valid class
        for class_name in valid_classes:
            class_path = os.path.join(self.data_dir, class_name)
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(class_path, f))]
            
            random.shuffle(images)

            n_total = len(images)
            n_test = int(n_total * test_split)
            n_val = int(n_total * val_split)

            splits = {
                "test": images[:n_test],
                "val": images[n_test:n_test+n_val],
                "train": images[n_test+n_val:]
            }

            print(f"📊 {class_name}: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

            for split, files in splits.items():
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                for f in files:
                    shutil.copyfile(os.path.join(class_path, f), os.path.join(split_class_dir, f))
        
        return output_dir

    # Data generators
    def setup_data_generators(self, split_dir):
        print(f"🔄 Setting up data generators from: {split_dir}")
        
        # Check if split directories exist and have content
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(split_dir, split)
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Split directory does not exist: {split_path}")
            
            # Count images in this split
            total_images = 0
            class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            
            for class_dir in class_dirs:
                class_path = os.path.join(split_path, class_dir)
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))])
                total_images += image_count
            
            print(f"📊 {split.upper()} split: {total_images} images in {len(class_dirs)} classes")
            
            if total_images == 0:
                raise ValueError(f"No images found in {split} split")
        
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

        print("🔄 Creating data generators...")
        
        try:
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
        except Exception as e:
            print(f"❌ Error creating data generators: {str(e)}")
            raise
        
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"✅ Classes found: {self.class_names}")
        print(f"📊 Class indices: {self.train_generator.class_indices}")

        # Compute class weights for imbalance (with error handling)
        try:
            labels = self.train_generator.classes
            if len(labels) == 0:
                raise ValueError("No training labels found")
            
            unique_labels = np.unique(labels)
            print(f"📊 Unique labels in training set: {unique_labels}")
            
            class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
            self.class_weights = dict(zip(unique_labels, class_weights))
            
            print(f"⚖️  Computed class weights: {self.class_weights}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not compute class weights: {str(e)}")
            print("Using equal weights for all classes")
            # Default to equal weights
            num_classes = len(self.class_names)
            self.class_weights = {i: 1.0 for i in range(num_classes)}

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
        
        # Initialize XAI visualizer
        self.xai_visualizer = XAIVisualizer(self.model, self.img_size)
        
        print(f"\n🔧 Model created with logit pruning ratio: {self.prune_ratio}")
        print(f"📊 Pruning layers will zero out lowest {self.prune_ratio*100:.0f}% of activations")
        print(f"🔍 XAI methods available: Saliency Maps, Grad-CAM, Occlusion")
        
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

        print("\n🔥 Phase 2: Fine-tuning last 50 layers with 30% Logit Pruning...")
        
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

        # Update XAI visualizer with fine-tuned model
        self.xai_visualizer.model = self.model

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
                
                print(f"\n🔍 {prune_layer_name}:")
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
            axes[1,1].text(0.1, 0.2, f"• XAI methods: Enabled", transform=axes[1,1].transAxes, fontsize=10)
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')

        plt.tight_layout()
        plt.suptitle('Enhanced GoogLeNet Training with 30% Logit Pruning & XAI', 
                     fontsize=16, y=1.02)
        plt.show()

    def run_xai_analysis(self, num_samples=5, save_dir="xai_results"):
        """
        Run XAI analysis on sample images from test set
        """
        print("\n🎯 Running XAI Analysis on Test Images")
        print("=" * 60)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get test images from each class
        test_dir = "split_dataset/test"
        sample_images = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                selected = random.sample(images, min(num_samples//2, len(images)))
                for img in selected:
                    sample_images.append((os.path.join(class_dir, img), class_name))
        
        # If we don't have enough, just take from any class
        if len(sample_images) < num_samples:
            all_test_images = []
            for class_name in self.class_names:
                class_dir = os.path.join(test_dir, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for img in images:
                        all_test_images.append((os.path.join(class_dir, img), class_name))
            
            additional_needed = num_samples - len(sample_images)
            additional_samples = random.sample(all_test_images, min(additional_needed, len(all_test_images)))
            sample_images.extend(additional_samples)
        
        # Run XAI analysis on each sample
        xai_results = []
        for i, (image_path, true_class) in enumerate(sample_images[:num_samples]):
            print(f"\n📸 Analyzing image {i+1}/{num_samples}: {os.path.basename(image_path)} (True class: {true_class})")
            
            save_path = os.path.join(save_dir, f"xai_analysis_{i+1}_{os.path.basename(image_path)}.png")
            
            try:
                result = self.xai_visualizer.visualize_xai_results(image_path, save_path)
                result['image_path'] = image_path
                result['true_class'] = true_class
                result['predicted_class'] = self.class_names[1] if result['prediction'] > 0.5 else self.class_names[0]
                xai_results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {image_path}: {str(e)}")
                continue
        
        # Summary of results
        self._print_xai_summary(xai_results)
        
        return xai_results
    
    def _print_xai_summary(self, xai_results):
        """Print summary of XAI analysis results"""
        print("\n📊 XAI Analysis Summary")
        print("=" * 40)
        
        correct_predictions = 0
        total_predictions = len(xai_results)
        
        for i, result in enumerate(xai_results, 1):
            is_correct = result['true_class'] == result['predicted_class']
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Image {i}: {os.path.basename(result['image_path'])}")
            print(f"   True: {result['true_class']} | Pred: {result['predicted_class']} ({result['prediction']:.3f})")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n🎯 Sample Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
        print(f"🔍 XAI visualizations saved in: xai_results/")

    # Enhanced evaluation with pruning analysis and XAI
    def evaluate_model(self, model_path="best_model_with_30_pruning.h5", run_xai=True):
        print("\n🎯 Evaluating Best Model with 30% Logit Pruning")
        print("=" * 50)
        
        # Load the best model with custom objects
        custom_objects = {
            "loss_fn": focal_loss(),
            "LogitPruningLayer": LogitPruningLayer
        }
        
        best_model = models.load_model(model_path, custom_objects=custom_objects)
        
        # Update XAI visualizer with best model
        self.xai_visualizer = XAIVisualizer(best_model, self.img_size)
        
        # Get predictions
        print("📊 Generating predictions...")
        preds = best_model.predict(self.test_generator, verbose=1)
        y_pred = (preds > 0.5).astype(int).ravel()
        y_true = self.test_generator.classes

        # Debug information
        print(f"🔍 Debug Info:")
        print(f"   Number of test samples: {len(y_true)}")
        print(f"   Unique true labels: {np.unique(y_true)}")
        print(f"   Unique predicted labels: {np.unique(y_pred)}")
        print(f"   Stored class names: {self.class_names}")
        print(f"   Test generator class indices: {self.test_generator.class_indices}")
        
        # Get actual class names from test generator
        actual_class_names = list(self.test_generator.class_indices.keys())
        print(f"   Actual class names from generator: {actual_class_names}")
        
        # Use only the classes that actually exist in the data
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [actual_class_names[i] for i in unique_labels if i < len(actual_class_names)]
        
        print(f"   Using target names: {target_names}")

        # Classification report with correct target names
        print("\n📋 Classification Report:")
        try:
            print(classification_report(y_true, y_pred, 
                                      labels=unique_labels,
                                      target_names=target_names))
        except Exception as e:
            print(f"❌ Error generating classification report: {e}")
            print("📊 Basic accuracy calculation:")
            accuracy = np.mean(y_pred == y_true)
            print(f"   Accuracy: {accuracy:.4f}")
            
            # Manual confusion matrix
            from collections import Counter
            print(f"\n📊 Prediction Distribution:")
            print(f"   True labels: {Counter(y_true)}")
            print(f"   Predicted labels: {Counter(y_pred)}")

        # Update class names for consistency
        self.class_names = actual_class_names

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
        
        # Run XAI analysis
        xai_results = None
        if run_xai:
            xai_results = self.run_xai_analysis(num_samples=5)
        
        return {
            'auc_score': auc_score,
            'accuracy': np.mean(y_pred == y_true),
            'predictions': preds,
            'true_labels': y_true,
            'xai_results': xai_results
        }

    def analyze_feature_importance_with_xai(self, image_path):
        """
        Detailed analysis of a single image using all XAI methods
        """
        print(f"\n🔬 Deep XAI Analysis: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # Load and preprocess image
        img_array, original_img = self.xai_visualizer.preprocess_image(image_path)
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        predicted_class = self.class_names[1] if prediction > 0.5 else self.class_names[0]
        
        print(f"📊 Model Prediction: {prediction:.4f} → {predicted_class}")
        print(f"🔍 Confidence: {max(prediction, 1-prediction):.2%}")
        
        # Compute all XAI methods
        print("\n🔄 Computing XAI methods...")
        saliency, _, _ = self.xai_visualizer.compute_saliency_map(image_path)
        grad_cam, _, _ = self.xai_visualizer.compute_grad_cam(image_path)
        occlusion, _, _ = self.xai_visualizer.compute_occlusion_map(image_path)
        
        # Statistical analysis of attention maps
        print("\n📈 Statistical Analysis of Attention:")
        print(f"Saliency Map - Mean: {saliency.mean():.3f}, Std: {saliency.std():.3f}, Max: {saliency.max():.3f}")
        print(f"Grad-CAM     - Mean: {grad_cam.mean():.3f}, Std: {grad_cam.std():.3f}, Max: {grad_cam.max():.3f}")
        print(f"Occlusion    - Mean: {occlusion.mean():.3f}, Std: {occlusion.std():.3f}, Max: {occlusion.max():.3f}")
        
        # Find most important regions
        saliency_top_10 = np.percentile(saliency, 90)
        gradcam_top_10 = np.percentile(grad_cam, 90)
        occlusion_top_10 = np.percentile(occlusion, 90)
        
        print(f"\n🎯 Top 10% Important Regions (threshold values):")
        print(f"Saliency: {saliency_top_10:.3f}")
        print(f"Grad-CAM: {gradcam_top_10:.3f}")
        print(f"Occlusion: {occlusion_top_10:.3f}")
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Row 1: Original and overlaid visualizations
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title(f'Original Image\nPred: {prediction:.3f} ({predicted_class})', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(original_img)
        axes[0, 1].imshow(saliency, alpha=0.5, cmap='hot')
        axes[0, 1].set_title('Saliency Overlay', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(original_img)
        axes[0, 2].imshow(grad_cam, alpha=0.5, cmap='jet')
        axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12)
        axes[0, 2].axis('off')
        
        # Row 2: Individual attention maps
        im1 = axes[1, 0].imshow(saliency, cmap='hot')
        axes[1, 0].set_title('Saliency Map', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        im2 = axes[1, 1].imshow(grad_cam, cmap='jet')
        axes[1, 1].set_title('Grad-CAM', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        im3 = axes[1, 2].imshow(occlusion, cmap='RdYlBu_r')
        axes[1, 2].set_title('Occlusion Map', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        # Row 3: Thresholded important regions
        saliency_thresh = saliency > saliency_top_10
        gradcam_thresh = grad_cam > gradcam_top_10
        occlusion_thresh = occlusion > occlusion_top_10
        
        axes[2, 0].imshow(original_img)
        axes[2, 0].imshow(saliency_thresh, alpha=0.7, cmap='Reds')
        axes[2, 0].set_title('Top 10% Saliency Regions', fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(original_img)
        axes[2, 1].imshow(gradcam_thresh, alpha=0.7, cmap='Reds')
        axes[2, 1].set_title('Top 10% Grad-CAM Regions', fontsize=12)
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(original_img)
        axes[2, 2].imshow(occlusion_thresh, alpha=0.7, cmap='Reds')
        axes[2, 2].set_title('Top 10% Occlusion Regions', fontsize=12)
        axes[2, 2].axis('off')
        
        plt.suptitle(f'Comprehensive XAI Analysis: {os.path.basename(image_path)}\n'
                    f'Model with 30% Logit Pruning', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()
        
        return {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'saliency': saliency,
            'grad_cam': grad_cam,
            'occlusion': occlusion,
            'saliency_stats': {'mean': saliency.mean(), 'std': saliency.std(), 'max': saliency.max()},
            'gradcam_stats': {'mean': grad_cam.mean(), 'std': grad_cam.std(), 'max': grad_cam.max()},
            'occlusion_stats': {'mean': occlusion.mean(), 'std': occlusion.std(), 'max': occlusion.max()}
        }


# -------------------------------------------------
# Main script with XAI integration
# -------------------------------------------------
if __name__ == "__main__":
    print("🧠 Enhanced GoogLeNet with 30% Logit Pruning & XAI Methods")
    print("=" * 70)
    print("🔍 XAI Methods: Saliency Maps, Grad-CAM, Occlusion Sensitivity")
    
    # Initialize classifier with 30% logit pruning
    classifier = EnhancedGoogLeNetClassifier(
        data_dir="/content/OvarianCancerDataset/OvarianCancerDataset",
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
    
    print("\nEvaluating final model with XAI analysis...")
    results = classifier.evaluate_model(run_xai=True)
    
    # Optional: Detailed XAI analysis on a specific image
    # Uncomment the following lines to run detailed analysis on a specific image
    # sample_image_path = "path/to/your/sample/image.jpg"
    # detailed_analysis = classifier.analyze_feature_importance_with_xai(sample_image_path)
    
    print(f"\n✅ Training Complete!")
    print(f"📊 Final AUC Score: {results['auc_score']:.4f}")
    print(f"📊 Final Accuracy: {results['accuracy']:.4f}")
    print(f"🔍 XAI Analysis: Complete")
    print(f"💾 Results saved in: xai_results/")
    print(f"🎯 Model features 30% logit pruning for efficiency")
    print(f"🔬 Explainability through Saliency, Grad-CAM & Occlusion")