# WITHOUT DROPOUT AND EARLY STOPPING

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
# Enhanced GoogLeNet (InceptionV3 backbone)
# -------------------------------------------------
class EnhancedGoogLeNetClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
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

    # Build model
    def create_model(self, learning_rate=1e-3, dropout1=0.5, dropout2=0.7):
        base_model = applications.InceptionV3(weights="imagenet", include_top=False,
                                              input_shape=(*self.img_size, 3))
        base_model.trainable = False  # freeze initially

        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout1)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout2)(x)
        predictions = layers.Dense(1, activation="sigmoid")(x)

        self.model = models.Model(inputs=base_model.input, outputs=predictions)

        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-4, nesterov=True)
        self.model.compile(optimizer=opt,
                           loss=focal_loss(alpha=0.75, gamma=2.0),
                           metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        return self.model

    # Train model (2-phase training)
    def train_model(self, epochs=100):
        callbacks = [
            EarlyStopping(monitor="val_auc", patience=25, restore_best_weights=True, min_delta=0.001),
            ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_auc", mode="max"),
            ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=10, min_lr=1e-7, verbose=1, mode="max")
        ]
        print("🚀 Phase 1: Feature Extraction (frozen base)")
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=30,  # Phase 1
            class_weight=self.class_weights,
            callbacks=callbacks
        )

        print("🔓 Phase 2: Fine-tuning last 50 layers...")
        for layer in self.model.layers[-50:]:
            if not isinstance(layer, layers.BatchNormalization):  
                layer.trainable = True

        self.model.compile(
            optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9, decay=1e-5),
            loss=focal_loss(alpha=0.75, gamma=2.0),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        fine_tune_epochs = epochs - 30
        print(f"⚡ Fine-tuning for {fine_tune_epochs} epochs...")
        self.history_fine = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=fine_tune_epochs,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        return self.history, self.history_fine

    # Plot curves
    def plot_training_history(self):
        acc = self.history.history['accuracy'] + self.history_fine.history['accuracy']
        val_acc = self.history.history['val_accuracy'] + self.history_fine.history['val_accuracy']
        auc = self.history.history['auc'] + self.history_fine.history['auc']
        val_auc = self.history.history['val_auc'] + self.history_fine.history['val_auc']

        epochs_range = range(len(acc))
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Acc')
        plt.plot(epochs_range, val_acc, label='Val Acc')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, auc, label='Train AUC')
        plt.plot(epochs_range, val_auc, label='Val AUC')
        plt.legend(loc='lower right')
        plt.title('AUC')

        plt.show()

    # Evaluate best model
    def evaluate_model(self, model_path="best_model.h5"):
        best_model = models.load_model(model_path, custom_objects={"loss_fn": focal_loss()})
        preds = best_model.predict(self.test_generator)
        y_pred = (preds > 0.5).astype(int).ravel()
        y_true = self.test_generator.classes

        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        auc_score = roc_auc_score(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, preds)
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.title("ROC Curve")
        plt.show()


# -------------------------------------------------
# Main script
# -------------------------------------------------
if __name__ == "__main__":
    classifier = EnhancedGoogLeNetClassifier(data_dir="/content/OvarianCancerDataset")
    split_dir = classifier.prepare_splits()
    classifier.setup_data_generators(split_dir)
    classifier.create_model()
    classifier.train_model(epochs=100)
    classifier.plot_training_history()
    classifier.evaluate_model()
