import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable()
class RefinementDecoder(keras.Model):
    def __init__(self, projection_dim, output_image_size, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.output_image_size = output_image_size
        filter_configs = [projection_dim, projection_dim // 2, projection_dim // 4]
        filter_configs = [max(f, 32) for f in filter_configs]
        self.decoder_blocks = []
        self.decoder_blocks.append(DecoderBlock(filter_configs[0], name="decoder_block_0"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[1], name="decoder_block_1"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[2], name="decoder_block_2"))
        self.final_conv = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid', name='output_image_conv')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        spatial_side = tf.cast(tf.sqrt(tf.cast(sequence_length, tf.float32)), tf.int32)
        tf.Assert(tf.equal(spatial_side * spatial_side, sequence_length),
                  ["Input sequence_length must be a perfect square for reshaping to 2D."])
        x = tf.reshape(inputs, (batch_size, spatial_side, spatial_side, self.projection_dim))
        for block in self.decoder_blocks:
            x = block(x)
        output_image = self.final_conv(x)
        return output_image

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "output_image_size": self.output_image_size,
        })
        return config
