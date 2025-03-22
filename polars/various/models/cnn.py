import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, LayerNormalization,
    MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate,
    Activation, Add
)
from .config import CNN_CONFIG

@register_keras_serializable()
class SqueezeExcitationBlock(tf.keras.layers.Layer):
    """
    Bloque Squeeze-and-Excitation como capa personalizada.
    """
    def __init__(self, filters: int, se_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio
        
        # Define layers
        self.gap = GlobalAveragePooling1D()
        self.dense1 = Dense(filters // se_ratio, activation='gelu')
        self.dense2 = Dense(filters, activation='sigmoid')
    
    def call(self, inputs):
        # Squeeze
        x = self.gap(inputs)
        
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Reshape for broadcasting
        x = tf.expand_dims(x, axis=1)
        
        # Scale
        return inputs * x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "se_ratio": self.se_ratio
        })
        return config

# Update the residual block to use the new layer
def create_residual_block(x, filters, dilation_rate=1):
    """
    Crea un bloque residual mejorado con dilated convolutions y SE.
    
    Parámetros:
    -----------
    x : tensor
        Tensor de entrada
    filters : int
        Número de filtros
    dilation_rate : int
        Tasa de dilatación para las convoluciones
        
    Retorna:
    --------
    tensor
        Tensor procesado
    """
    skip = x
    
    # Convolution path
    x = Conv1D(
        filters=filters,
        kernel_size=CNN_CONFIG['kernel_size'],
        padding='same',
        dilation_rate=dilation_rate
    )(x)
    x = LayerNormalization()(x)
    x = Activation(CNN_CONFIG['activation'])(x)
    
    # Squeeze-and-Excitation
    if CNN_CONFIG['use_se_block']:
        x = SqueezeExcitationBlock(filters, CNN_CONFIG['se_ratio'])(x)
    
    # Project residual if needed
    if skip.shape[-1] != filters:
        skip = Conv1D(filters, 1, padding='same')(skip)
    
    return Add()([x, skip])

def create_cnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo CNN (Convolutional Neural Network) con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo CNN compilado
    """
    # Entrada CGM
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    
    # Proyección inicial
    x = Conv1D(CNN_CONFIG['filters'][0], 1, padding='same')(cgm_input)
    x = LayerNormalization()(x) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(x)
    
    # Bloques residuales con different dilation rates
    for filters in CNN_CONFIG['filters']:
        for dilation_rate in CNN_CONFIG['dilation_rates']:
            x = create_residual_block(x, filters, dilation_rate)
        x = MaxPooling1D(pool_size=CNN_CONFIG['pool_size'])(x)
    
    # Pooling global con concat de max y average
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Entrada de otras características
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Combinar características
    combined = Concatenate()([x, other_input])
    
    # Capas densas con residual connections
    skip = combined
    dense = Dense(256, activation=CNN_CONFIG['activation'])(combined)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'])(dense)
    dense = Dense(256, activation=CNN_CONFIG['activation'])(dense)
    if skip.shape[-1] == 256:
        dense = Add()([dense, skip])
    
    # Final layers
    dense = Dense(128, activation=CNN_CONFIG['activation'])(dense)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'] / 2)(dense)
    
    output = Dense(1)(dense)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)

def create_cnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo CNN (Convolutional Neural Network) con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo CNN compilado
    """
    # Entrada CGM
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    
    # Proyección inicial
    x = Conv1D(CNN_CONFIG['filters'][0], 1, padding='same')(cgm_input)
    x = LayerNormalization()(x) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(x)
    
    # Bloques residuales con different dilation rates
    for filters in CNN_CONFIG['filters']:
        for dilation_rate in CNN_CONFIG['dilation_rates']:
            x = create_residual_block(x, filters, dilation_rate)
        x = MaxPooling1D(pool_size=CNN_CONFIG['pool_size'])(x)
    
    # Pooling global con concat de max y average
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Entrada de otras características
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Combinar características
    combined = Concatenate()([x, other_input])
    
    # Capas densas con residual connections
    skip = combined
    dense = Dense(256, activation=CNN_CONFIG['activation'])(combined)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'])(dense)
    dense = Dense(256, activation=CNN_CONFIG['activation'])(dense)
    if skip.shape[-1] == 256:
        dense = Add()([dense, skip])
    
    # Final layers
    dense = Dense(128, activation=CNN_CONFIG['activation'])(dense)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'] / 2)(dense)
    
    output = Dense(1)(dense)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)