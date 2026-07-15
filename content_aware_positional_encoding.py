import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable()
class ContentAwarePositionalEncoding(layers.Layer):
    def __init__(self, target_spatial_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.target_spatial_size = target_spatial_size
        self.projection_dim = projection_dim
        self.conv1x1 = layers.Conv2D(projection_dim, kernel_size=1, activation='gelu', name='cape_conv1x1')

    def call(self, image_features, output_sequence_length):
        batch_size = tf.shape(image_features)[0]
        pooled_features = tf.image.resize(
            image_features,
            size=(self.target_spatial_size, self.target_spatial_size),
            method=tf.image.ResizeMethod.BILINEAR
        )
        cape_representation = self.conv1x1(pooled_features)
        output_spatial_side = tf.cast(tf.sqrt(tf.cast(output_sequence_length, tf.float32)), tf.int32)
        tf.Assert(tf.equal(output_spatial_side * output_spatial_side, output_sequence_length),
                  ["Input output_sequence_length must be a perfect square for reshaping to a 2D grid."])

        rescaled_cape = tf.image.resize(
            cape_representation,
            size=(output_spatial_side, output_spatial_side),
            method=tf.image.ResizeMethod.BILINEAR
        )
        final_cape_encoding = tf.reshape(
            rescaled_cape,
            (batch_size, output_sequence_length, self.projection_dim)
        )
        return final_cape_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_spatial_size": self.target_spatial_size,
            "projection_dim": self.projection_dim,
        })
        return config
