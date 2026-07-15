import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable()
class DecoderBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        # Main convolutional path for the residual block
        # Using SeparableConv2D as per "Xception-like" principles
        self.res_conv1 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        self.res_bn1 = layers.BatchNormalization()
        self.res_act1 = layers.Activation('relu') # ReLU is a common choice for decoder activations

        self.res_conv2 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        self.res_bn2 = layers.BatchNormalization()

        # Shortcut path for residual connection, initialized in build if channel matching is needed
        self.shortcut_conv = None

        # Upsampling operation, as specified to be part of each stage
        self.upsample_layer = layers.UpSampling2D(size=2, interpolation='bilinear')

    def build(self, input_shape):
        # Create a 1x1 convolution for the shortcut if input channels don't match the block's output filters
        if input_shape[-1] != self.filters:
            self.shortcut_conv = keras.Sequential([
                layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False),
                layers.BatchNormalization()
            ], name='shortcut_conv_for_channel_matching')
        super().build(input_shape)

    def call(self, inputs):
        # Store inputs for the residual connection
        residual = inputs

        # Main convolutional path
        x = self.res_conv1(inputs)
        x = self.res_bn1(x)
        x = self.res_act1(x)

        x = self.res_conv2(x)
        x = self.res_bn2(x)

        # Apply shortcut transformation if necessary
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)

        # Add residual and main paths, then activate
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x) # Final activation after residual addition

        # Perform upsampling as the last step of the stage
        x = self.upsample_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config
