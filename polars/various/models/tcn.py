import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, LayerNormalization,
    BatchNormalization, Add, Activation, GlobalAveragePooling1D
)
from .config import TCN_CONFIG

class CausalPadding(tf.keras.layers.Layer):
    """
    Capa personalizada para padding causal.
    """
    def __init__(self, padding_size, **kwargs):
        super().__init__(**kwargs)
        self.padding_size = padding_size

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding_size, 0], [0, 0]])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.padding_size, input_shape[2])

def create_tcn_block(input_layer: tf.Tensor, filters: int, kernel_size: int, 
                    dilation_rate: int, dropout_rate: float) -> tf.Tensor:
    """
    Crea un bloque TCN (Temporal Convolutional Network).
    
    Parámetros:
    -----------
    input_layer : tf.Tensor
        Capa de entrada
    filters : int
        Número de filtros
    kernel_size : int
        Tamaño del kernel
    dilation_rate : int
        Tasa de dilatación
    dropout_rate : float
        Tasa de dropout
    
    Retorna:
    --------
    tf.Tensor
        Salida del bloque TCN
    """
    # Padding causal para mantener causalidad temporal
    padding_size = (kernel_size - 1) * dilation_rate
    padded_input = CausalPadding(padding_size)(input_layer)
    
    # Convolución dilatada
    conv = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='valid',
        activation='relu'
    )(padded_input)
    
    # Normalización y regularización
    conv = LayerNormalization(epsilon=TCN_CONFIG['epsilon'])(conv)
    conv = Dropout(dropout_rate)(conv)
    
    # Conexión residual si las dimensiones coinciden
    if input_layer.shape[-1] == filters:
        cropped_input = input_layer[:, -conv.shape[1]:, :]
        return Add()([conv, cropped_input])
    return conv

def create_tcn_model(input_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo TCN completo.
    
    Parámetros:
    -----------
    input_shape : tuple
        Forma de los datos CGM
    other_features_shape : tuple
        Forma de otras características
    
    Retorna:
    --------
    Model
        Modelo TCN compilado
    """
    # Entradas
    cgm_input = Input(shape=input_shape[1:], name='cgm_input')
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Bloques TCN
    x = cgm_input
    skip_connections = []
    
    for dilation_rate in TCN_CONFIG['dilations']:
        tcn_out = create_tcn_block(
            x,
            filters=TCN_CONFIG['filters'],
            kernel_size=TCN_CONFIG['kernel_size'],
            dilation_rate=dilation_rate,
            dropout_rate=TCN_CONFIG['dropout_rate'][0]  # Using first dropout rate for TCN blocks
        )
        skip_connections.append(tcn_out)
        x = tcn_out
    
    # Combinar skip connections
    if skip_connections:
        target_len = skip_connections[-1].shape[1]
        aligned_skips = [
            skip[:, -target_len:, :] for skip in skip_connections
        ]
        x = Add()(aligned_skips)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Combinar con otras características
    x = tf.keras.layers.Concatenate()([x, other_input])
    
    # Capas densas finales
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][0])(x)  # First dropout rate
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][1])(x)  # Second dropout rate
    
    output = Dense(1, activation='linear')(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)