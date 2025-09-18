import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_dla_block(x, filters, kernel_size=3, stride=1, name_prefix=""):
    """
    Create a DLA (Deep Layer Aggregation) block
    """
    # Main path
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                     name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                     name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                               name=f'{name_prefix}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
    
    # Add shortcut connection
    x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
    
    return x

class ResizeLayer(layers.Layer):
    """Custom layer to resize feature maps to match target dimensions"""
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
    
    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])
    
    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

def create_dla_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Create Deep Layer Aggregation (DLA) model for binary classification
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='initial_conv')(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(name='initial_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='initial_pool')(x)
    
    # DLA blocks with hierarchical feature aggregation
    # Level 1 - Size: H/4 x W/4
    x1 = create_dla_block(x, 64, name_prefix='level1_block1')
    x1 = create_dla_block(x1, 64, name_prefix='level1_block2')
    
    # Level 2 - Size: H/8 x W/8  
    x2 = create_dla_block(x1, 128, stride=2, name_prefix='level2_block1')
    x2 = create_dla_block(x2, 128, name_prefix='level2_block2')
    
    # Level 3 - Size: H/16 x W/16
    x3 = create_dla_block(x2, 256, stride=2, name_prefix='level3_block1')
    x3 = create_dla_block(x3, 256, name_prefix='level3_block2')
    x3 = create_dla_block(x3, 256, name_prefix='level3_block3')
    
    # Level 4 - Size: H/32 x W/32
    x4 = create_dla_block(x3, 512, stride=2, name_prefix='level4_block1')
    x4 = create_dla_block(x4, 512, name_prefix='level4_block2')
    x4 = create_dla_block(x4, 512, name_prefix='level4_block3')
    
    # Hierarchical Feature Aggregation (HFA)
    # Calculate expected dimensions based on input size and pooling
    # After initial conv (stride=2) and maxpool (stride=2): input/4
    # Then level2 (stride=2): input/8 - this is our target size
    expected_h, expected_w = input_shape[0] // 8, input_shape[1] // 8
    
    # Upsample x3 and x4 to match x2 dimensions
    x3_up = layers.UpSampling2D(2, name='x3_upsample')(x3)  # H/16 -> H/8
    x4_up = layers.UpSampling2D(4, name='x4_upsample')(x4)  # H/32 -> H/8
    
    # Use custom resize layer to ensure exact dimension matching
    x3_up = ResizeLayer(expected_h, expected_w, name='x3_resize')(x3_up)
    x4_up = ResizeLayer(expected_h, expected_w, name='x4_resize')(x4_up)
    
    # Reduce channel dimensions to make concatenation manageable
    x2_reduced = layers.Conv2D(128, 1, padding='same', name='x2_reduce')(x2)
    x3_reduced = layers.Conv2D(128, 1, padding='same', name='x3_reduce')(x3_up)
    x4_reduced = layers.Conv2D(128, 1, padding='same', name='x4_reduce')(x4_up)
    
    # Aggregate features
    aggregated = layers.Concatenate(name='feature_aggregation')([x2_reduced, x3_reduced, x4_reduced])
    
    # Feature fusion
    fusion = layers.Conv2D(512, 3, padding='same', name='fusion_conv')(aggregated)
    fusion = layers.BatchNormalization(name='fusion_bn')(fusion)
    fusion = layers.ReLU(name='fusion_relu')(fusion)
    
    # Additional processing
    fusion = layers.Conv2D(256, 3, padding='same', name='fusion_conv2')(fusion)
    fusion = layers.BatchNormalization(name='fusion_bn2')(fusion)
    fusion = layers.ReLU(name='fusion_relu2')(fusion)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(fusion)
    
    # Dropout for regularization
    x = layers.Dropout(0.5, name='dropout')(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs, outputs, name='DLA_OvarianCancer')
    return model

def prepare_data_generators(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Prepare data generators with augmentation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Raw_Annotated_Samples'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'Raw_Annotated_Samples'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

def train_dla_model(data_dir, epochs=20):
    """
    Train the DLA model
    """
    # Create data generators
    train_gen, val_gen = prepare_data_generators(data_dir)
    
    print(f"Found {train_gen.samples} training samples")
    print(f"Found {val_gen.samples} validation samples")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Create model
    model = create_dla_model()
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(name='accuracy'), 
                Precision(name='precision'), 
                Recall(name='recall')]
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_dla_ovarian_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_generator):
    """
    Evaluate the trained model
    """
    # Evaluate on validation set
    results = model.evaluate(val_generator, verbose=0)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    print(f"Validation Precision: {results[2]:.4f}")
    print(f"Validation Recall: {results[3]:.4f}")
    
    # Calculate F1 score
    f1_score = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"Validation F1-Score: {f1_score:.4f}")
    print("="*50)

# Main execution
if __name__ == "__main__":
    # Set your data directory path
    DATA_DIR = "ovarian_data"  # Update this to your actual data path
    
    print("Starting DLA Model Training for Ovarian Cancer Classification")
    print("="*60)
    
    # Train model
    model, history = train_dla_model(DATA_DIR, epochs=20)
    
    # Plot training history
    plot_training_history(history)
    
    # Create validation generator for evaluation
    _, val_gen = prepare_data_generators(DATA_DIR)
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    print("\nTraining completed!")
    print("Best model saved as 'best_dla_ovarian_model.h5'")

# Example of how to make predictions on new images
def predict_single_image(model, image_path, img_size=(224, 224)):
    """
    Predict on a single image
    """
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Convert to class label
    class_names = ['Benign', 'Malignant']
    predicted_class = class_names[int(prediction[0] > 0.5)]
    confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_class, confidence

# Example usage for prediction:
# loaded_model = tf.keras.models.load_model('best_dla_ovarian_model.h5')
# predict_single_image(loaded_model, 'path/to/your/image.jpg')