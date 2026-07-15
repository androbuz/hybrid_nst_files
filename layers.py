import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layers = keras.Sequential(
            [
                layers.Dense(ffn_units, activation="gelu"),
                layers.Dense(embed_dim),
            ]
        )
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        norm1 = self.layernorm1(inputs)
        attn_output = self.att(query=norm1, value=norm1, key=norm1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        norm2 = self.layernorm2(out1)
        ffn_output = self.ffn_layers(norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
        })
        return config
        

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
        

class CLIPPatchLoss(keras.layers.Layer):
    def __init__(self, target_text_embedding, num_patches=64, patch_size=128, image_size=256, **kwargs):
        super().__init__(**kwargs)
        self.target_text_embedding = target_text_embedding
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size

        # Define augmentations for the patches
        # Using a Keras Sequential model for augmentations
        self.augmenter = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(factor=0.05), # 5% rotation
                keras.layers.RandomZoom(
                    height_factor=(-0.3, 0.2),
                    width_factor=(-0.3, 0.2)
                ),
                # For more 'perspective' like transforms, tf.image.transform with custom matrices would be needed.
                # For now, these basic affine transforms should provide some robustness.
            ],
            name="patch_augmenter",
        )

    def call(self, generated_image, training=False):
        # Ensure generated_image is in [0, 1] range for processing
        generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)

        batch_size = tf.shape(generated_image)[0]

        # Replicate each image `self.num_patches` times to get multiple crops per image
        # Shape becomes (batch_size * num_patches, H, W, C)
        repeated_images = tf.repeat(generated_image, repeats=self.num_patches, axis=0)

        # Generate `batch_size * num_patches` random crops in a single vectorized operation
        cropped_patches = tf.image.random_crop(
            repeated_images,
            size=[batch_size * self.num_patches, self.patch_size, self.patch_size, 3]
        )

        # Apply augmentations to all cropped patches
        augmented_patches = self.augmenter(cropped_patches, training=training)

        # Keras CV Image Encoder expects images in [0, 255] range as float32
        processed_image_input = tf.cast(augmented_patches * 255.0, dtype=tf.float32)

        # Resize image to the expected input size for CLIP image encoder (224x224 for ViT-B/32)
        clip_image_input_size = 224 # Standard input size for CLIP ViT-B/32
        resized_image_input = tf.image.resize(
            processed_image_input,
            (clip_image_input_size, clip_image_input_size)
        )

        # Use the Keras CV image encoder for image encoding.
        image_features = clip_image_encoder(resized_image_input)

        # Normalize the image features
        image_embedding = tf.linalg.normalize(image_features)[0]

        # `image_embedding` now contains all patch embeddings: (batch_size * num_patches, embed_dim)
        stacked_patch_embeddings = image_embedding

        # Expand text embedding to match the batch size of stacked_patch_embeddings
        expanded_target_text_embedding = tf.tile(
            self.target_text_embedding,
            [tf.shape(stacked_patch_embeddings)[0], 1]
        )

        # Compute cosine similarity.
        # Maximizing similarity is equivalent to minimizing negative similarity.
        cosine_sim = tf.reduce_sum(expanded_target_text_embedding * stacked_patch_embeddings, axis=-1)
        clip_loss = -tf.reduce_mean(cosine_sim)

        return clip_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                # target_text_embedding is a Tensor, so it needs to be serialized manually or passed during instantiation
                # For this example, let's assume it's set at runtime or won't be saved directly via get_config.
            }
        )
        return config
