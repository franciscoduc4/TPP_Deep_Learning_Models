import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
from .config import ATTENTION_CONFIG

def create_attention_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout_rate: float) -> tf.Tensor:
    """
    Crea un bloque de atención con feed-forward network.

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
    
    Retorna:
    --------
    tf.Tensor
        Tensor de salida del bloque de atención
    """
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    
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
    
    x = cgm_input
    
    # Stack attention blocks
    for _ in range(ATTENTION_CONFIG['num_layers']):
        x = create_attention_block(
            x,
            ATTENTION_CONFIG['num_heads'],
            ATTENTION_CONFIG['key_dim'],
            ATTENTION_CONFIG['ff_dim'],
            ATTENTION_CONFIG['dropout_rate']
        )
    
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, other_input])
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(ATTENTION_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)