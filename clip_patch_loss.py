import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
