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
