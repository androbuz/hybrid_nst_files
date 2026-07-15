import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        # Reshape layer to flatten the patches from (h, w, c*p*p) to (-1, c*p*p)
        # Assuming 3 channels for images
        self.flatten_patches = layers.Reshape((-1, (patch_size * patch_size * 3)))
        self.projection = layers.Dense(projection_dim)

    def call(self, images):
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = self.flatten_patches(patches)
        projected_patches = self.projection(patches)
        return projected_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
        })
        return config
