import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, Activation,
    BatchNormalization, GlobalAveragePooling1D, Concatenate
)
from .config import WAVENET_CONFIG

class WaveNetBlock(tf.keras.layers.Layer):
    """
    Bloque WaveNet personalizado.
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal'
        )
        self.bn = BatchNormalization()
        self.activation = Activation('relu')
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.residual_proj = Conv1D(filters, 1, padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Ensure residual has same number of filters
        residual = self.residual_proj(inputs)
        
        # Match temporal dimension
        residual = residual[:, -x.shape[1]:, :]
        return self.add([x, residual])

def create_wavenet_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """
    Crea un bloque WaveNet con conexiones residuales y skip connections.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    filters : int
        Número de filtros de la capa convolucional
    kernel_size : int
        Tamaño del kernel de la capa convolucional
    dilation_rate : int
        Tasa de dilatación de la capa convolucional
    dropout_rate : float
        Tasa de dropout

    Retorna:
    --------
    tf.Tensor
        Tensor de salida del bloque WaveNet
    """
    # Convolución dilatada
    conv = Conv1D(filters=filters, kernel_size=kernel_size,
                 dilation_rate=dilation_rate, padding='causal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(dropout_rate)(conv)
    
    # Conexión residual con proyección 1x1 si es necesario
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    
    # Alinear dimensiones temporales
    x = x[:, -conv.shape[1]:, :]
    res = Add()([conv, x])
    
    return res, conv

def create_wavenet_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo WaveNet para predicción de series temporales.

    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)

    Retorna:
    --------
    Model
        Modelo WaveNet compilado
    """
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    x = Conv1D(WAVENET_CONFIG['filters'][0], 1, padding='same')(cgm_input)
    current_filters = WAVENET_CONFIG['filters'][0]
    
    skip_outputs = []
    
    for filters in WAVENET_CONFIG['filters']:
        for dilation in WAVENET_CONFIG['dilations']:
            wavenet_block = WaveNetBlock(
                filters=filters,
                kernel_size=WAVENET_CONFIG['kernel_size'],
                dilation_rate=dilation,
                dropout_rate=WAVENET_CONFIG['dropout_rate']
            )
            x = wavenet_block(x)
            
            # Project skip connection to match final filter size
            skip_proj = Conv1D(WAVENET_CONFIG['filters'][-1], 1, padding='same')(x)
            skip_outputs.append(skip_proj)
    
    # Combinar skip connections
    if skip_outputs:
        target_len = skip_outputs[-1].shape[1]
        aligned_skips = [
            skip[:, -target_len:, :] for skip in skip_outputs
        ]
        x = Add()(aligned_skips)
    
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Concatenate()([x, other_input])
    
    # Combinar con otras características
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(WAVENET_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)