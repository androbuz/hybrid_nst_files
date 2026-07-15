import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.saving.register_keras_serializable()
class StyleTransferModel(keras.Model):
    def __init__(self,
                 content_patch_embedder,
                 style_patch_embedder,
                 content_encoder,
                 style_encoder,
                 cape_layer,
                 refinement_decoder,
                 projection_dim,
                 style_text_embedding_dim,
                 num_patches,
                 **kwargs):
        super().__init__(**kwargs)
        self.content_patch_embedder = content_patch_embedder
        self.style_patch_embedder = style_patch_embedder
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.cape_layer = cape_layer
        self.refinement_decoder = refinement_decoder
        self.projection_dim = projection_dim
        self.style_text_embedding_dim = style_text_embedding_dim
        self.num_patches = num_patches

        self.combine_features_layer = layers.Dense(self.projection_dim, name='combine_features')
        self.text_to_style_projection = layers.Dense(self.num_patches * self.projection_dim,
                                                     name="text_to_style_projection_layer")

    def call(self, content_img, style_img=None, style_text_embedding=None, training=False):
        content_patches = self.content_patch_embedder(content_img)
        output_sequence_length = tf.shape(content_patches)[1]
        content_cape_encoding = self.cape_layer(content_img, output_sequence_length)
        content_patches_with_cape = content_patches + content_cape_encoding
        encoded_content = self.content_encoder(content_patches_with_cape)

        if style_img is not None:
            style_patches = self.style_patch_embedder(style_img)
            style_cape_encoding = self.cape_layer(style_img, output_sequence_length)
            style_patches_with_cape = style_patches + style_cape_encoding
            encoded_style = self.style_encoder(style_patches_with_cape)
        else:
            target_batch_size = tf.shape(content_img)[0]
            tiled_style_text_embedding = tf.tile(style_text_embedding, [target_batch_size, 1])
            projected_text_style = self.text_to_style_projection(tiled_style_text_embedding)
            encoded_style = tf.reshape(projected_text_style, [target_batch_size, self.num_patches, self.projection_dim])

        decoder_input = tf.concat([encoded_content, encoded_style], axis=-1)
        combined_features = self.combine_features_layer(decoder_input)
        generated_image = self.refinement_decoder(combined_features)
        return generated_image

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "style_text_embedding_dim": self.style_text_embedding_dim,
            "num_patches": self.num_patches,
            "content_patch_embedder": self.content_patch_embedder,
            "style_patch_embedder": self.style_patch_embedder,
            "content_encoder": self.content_encoder,
            "style_encoder": self.style_encoder,
            "cape_layer": self.cape_layer,
            "refinement_decoder": self.refinement_decoder,
        })
        return config
