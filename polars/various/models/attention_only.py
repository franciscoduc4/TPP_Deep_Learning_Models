import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
    Add
)
from .config import ATTENTION_CONFIG

class RelativePositionEncoding(tf.keras.layers.Layer):
    """
    Codificación de posición relativa para mejorar la atención temporal.
    """
    def __init__(self, max_position: int, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.depth = depth
        
    def build(self, input_shape):
        self.rel_embeddings = self.add_weight(
            name="rel_embeddings",
            shape=[2 * self.max_position - 1, self.depth],
            initializer="glorot_uniform"
        )
        
    def call(self, length):
        pos_emb = tf.gather(
            self.rel_embeddings,
            tf.range(length)[:, tf.newaxis] - tf.range(length)[tf.newaxis, :] + self.max_position - 1
        )
        return pos_emb

def create_attention_block(x: tf.Tensor, num_heads: int, key_dim: int, 
                         ff_dim: int, dropout_rate: float, training: bool = None) -> tf.Tensor:
    """
    Crea un bloque de atención mejorado con posición relativa y gating.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de la clave
    ff_dim : int
        Dimensión de la red feed-forward
    dropout_rate : float
        Tasa de dropout
    training : bool
        Indica si está en modo entrenamiento
    
    Retorna:
    --------
    tf.Tensor
        Tensor procesado
    """
    # Relative position encoding
    if ATTENTION_CONFIG['use_relative_attention']:
        pos_encoding = RelativePositionEncoding(
            ATTENTION_CONFIG['max_relative_position'],
            key_dim
        )(tf.shape(x)[1])
        
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=ATTENTION_CONFIG['head_size']
        )(x, x, attention_bias=pos_encoding)
    else:
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )(x, x)
    
    # Gating mechanism
    gate = tf.keras.layers.Dense(attention_output.shape[-1], activation='sigmoid')(x)
    attention_output = gate * attention_output
    
    attention_output = Dropout(dropout_rate)(attention_output, training=training)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Enhanced feed-forward network with GLU
    ffn = Dense(ff_dim)(x)
    ffn_gate = Dense(ff_dim, activation='sigmoid')(x)
    ffn = ffn * ffn_gate
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn, training=training)
    
    return LayerNormalization(epsilon=1e-6)(x + ffn)

def create_attention_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo basado únicamente en mecanismos de atención.

    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)

    Retorna:
    --------
    Model
        Modelo de atención compilado
    """
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Initial projection
    x = Dense(ATTENTION_CONFIG['key_dim'] * ATTENTION_CONFIG['num_heads'])(cgm_input)
    
    # Stochastic depth (layer dropout)
    survive_rates = tf.linspace(1.0, 0.5, ATTENTION_CONFIG['num_layers'])
    
    # Stack attention blocks with stochastic depth
    for i in range(ATTENTION_CONFIG['num_layers']):
        if tf.random.uniform([]) < survive_rates[i]:
            x = create_attention_block(
                x,
                ATTENTION_CONFIG['num_heads'],
                ATTENTION_CONFIG['key_dim'],
                ATTENTION_CONFIG['ff_dim'],
                ATTENTION_CONFIG['dropout_rate']
            )
    
    # Global context
    attention_pooled = GlobalAveragePooling1D()(x)
    max_pooled = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = Concatenate()([attention_pooled, max_pooled])
    
    # Combine with other features
    x = Concatenate()([x, other_input])
    
    # Final MLP with residual
    skip = x
    x = Dense(128, activation=ATTENTION_CONFIG['activation'])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(ATTENTION_CONFIG['dropout_rate'])(x)
    x = Dense(128, activation=ATTENTION_CONFIG['activation'])(x)
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)