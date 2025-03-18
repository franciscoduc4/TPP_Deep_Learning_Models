import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Concatenate, Activation
)
from .config import TABNET_CONFIG

class GLU(tf.keras.layers.Layer):
    """
    Gated Linear Unit como capa personalizada.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = Dense(units * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        return x[:, :self.units] * tf.nn.sigmoid(x[:, self.units:])

class FeatureTransformer(tf.keras.layers.Layer):
    """
    Transformador de características como capa personalizada.
    """
    def __init__(self, feature_dim, batch_momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.glu = GLU(feature_dim)
        self.bn = BatchNormalization(momentum=batch_momentum)

    def call(self, inputs):
        x = self.glu(inputs)
        return self.bn(x)

def custom_softmax(x: tf.Tensor, axis: int=-1) -> tf.Tensor:
    """
    Implementación de softmax con estabilidad numérica.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    axis : int
        Eje de normalización
    
    Retorna:
    --------
    tf.Tensor
        Tensor normal
    """
    exp_x = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=axis, keepdims=True)

def glu(x: tf.Tensor, n_units: int) -> tf.Tensor:
    """
    Gated Linear Unit.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    n_units : int
        Número de unidades

    Retorna:
    --------
    tf.Tensor
        Tensor GLU
    """
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])

def feature_transformer(x: tf.Tensor, feature_dim: int, batch_momentum: float=0.98) -> tf.Tensor:
    """
    Transformador de características.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    feature_dim : int
        Dimensión de las características
    batch_momentum : float
        Momento de la normalización por lotes
    
    Retorna:
    --------
    tf.Tensor
        Tensor transform
    """
    transform = Dense(feature_dim * 2)(x)
    transform = glu(transform, feature_dim)
    return BatchNormalization(momentum=batch_momentum)(transform)

def create_tabnet_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo TabNet modificado para procesamiento de datos tabulares.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM
    other_features_shape : tuple
        Forma de otras características
        
    Retorna:
    --------
    Model
        Modelo TabNet compilado
    """
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    cgm_flat = tf.keras.layers.Flatten()(cgm_input)
    x = Concatenate()([cgm_flat, other_input])
    
    for _ in range(TABNET_CONFIG['num_decision_steps']):
        transformer = FeatureTransformer(
            TABNET_CONFIG['feature_dim'],
            TABNET_CONFIG['batch_momentum']
        )
        x = transformer(x)
        
        mask = Dense(x.shape[-1], activation='softmax')(x)
        x = tf.keras.layers.Multiply()([x, mask])
    
    # Final layers
    x = Dense(TABNET_CONFIG['output_dim'], activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)