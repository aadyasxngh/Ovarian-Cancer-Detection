import tensorflow as tf
from tensorflow.keras import layers, models

class OptimizedGoogLeNetClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=16):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []

    def inception_module(self, x, f1, f3_in, f3_out, f5_in, f5_out, pool_proj):
        conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

        conv3 = layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(x)
        conv3 = layers.Conv2D(f3_out, (3, 3), padding='same', activation='relu')(conv3)

        conv5 = layers.Conv2D(f5_in, (1, 1), padding='same', activation='relu')(x)
        conv5 = layers.Conv2D(f5_out, (5, 5), padding='same', activation='relu')(conv5)

        pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = layers.Conv2D(pool_proj, (1, 1), padding='same', activation='relu')(pool)

        return layers.concatenate([conv1, conv3, conv5, pool], axis=-1)

    def create_model(self, learning_rate=1e-4):
        input_layer = layers.Input(shape=(*self.img_size, 3))

        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = self.inception_module(x, 64, 96, 128, 16, 32, 32)
        x = self.inception_module(x, 128, 128, 192, 32, 96, 64)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = self.inception_module(x, 192, 96, 208, 16, 48, 64)
        x = self.inception_module(x, 160, 112, 224, 24, 64, 64)
        x = self.inception_module(x, 128, 128, 256, 24, 64, 64)
        x = self.inception_module(x, 112, 144, 288, 32, 64, 64)
        x = self.inception_module(x, 256, 160, 320, 32, 128, 128)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = self.inception_module(x, 256, 160, 320, 32, 128, 128)
        x = self.inception_module(x, 384, 192, 384, 48, 128, 128)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        output_layer = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )

        print("GoogLeNet model created successfully!")
        print(f"Total parameters: {model.count_params():,}")

        return model
