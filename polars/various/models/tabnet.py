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

class MultiHeadFeatureAttention(tf.keras.layers.Layer):
    """
    Atención multi-cabeza para características.
    """
    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None):
        attention_output = self.attention(inputs, inputs, training=training)
        return self.layernorm(inputs + attention_output)

class EnhancedFeatureTransformer(tf.keras.layers.Layer):
    """
    Transformador de características mejorado con atención y ghost batch norm.
    """
    def __init__(self, feature_dim: int, num_heads: int, 
                 virtual_batch_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.virtual_batch_size = virtual_batch_size
        
        # GLU layers
        self.glu1 = GLU(feature_dim)
        self.glu2 = GLU(feature_dim)
        
        # Attention layer
        self.attention = MultiHeadFeatureAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Ghost Batch Normalization
        self.ghost_bn1 = tf.keras.layers.BatchNormalization(
            virtual_batch_size=virtual_batch_size
        )
        self.ghost_bn2 = tf.keras.layers.BatchNormalization(
            virtual_batch_size=virtual_batch_size
        )
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.glu1(inputs)
        x = self.ghost_bn1(x, training=training)
        x = self.attention(x, training=training)
        x = self.glu2(x)
        x = self.ghost_bn2(x, training=training)
        return self.dropout(x, training=training)

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


class TabNetModel(tf.keras.Model):
    """
    Modelo TabNet personalizado con manejo de pérdidas de entropía.
    """
    def __init__(self, cgm_shape, other_features_shape, **kwargs):
        super().__init__(**kwargs)
        self.cgm_shape = cgm_shape
        self.other_shape = other_features_shape
        self.entropy_tracker = tf.keras.metrics.Mean(name='entropy_loss')
        
        # Definir capas
        self.flatten = tf.keras.layers.Flatten()
        self.feature_dropout = tf.keras.layers.Dropout(TABNET_CONFIG['feature_dropout'])
        self.transformers = [
            EnhancedFeatureTransformer(
                feature_dim=TABNET_CONFIG['feature_dim'],
                num_heads=TABNET_CONFIG['num_attention_heads'],
                virtual_batch_size=TABNET_CONFIG['virtual_batch_size'],
                dropout_rate=TABNET_CONFIG['attention_dropout']
            ) for _ in range(TABNET_CONFIG['num_decision_steps'])
        ]
        
        # Capas finales
        self.final_dense1 = Dense(TABNET_CONFIG['output_dim'], activation='selu')
        self.final_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_dropout = tf.keras.layers.Dropout(TABNET_CONFIG['attention_dropout'])
        self.final_dense2 = Dense(TABNET_CONFIG['output_dim'] // 2, activation='selu')
        self.final_norm2 = tf.keras.layers.LayerNormalization()
        self.final_dense3 = Dense(TABNET_CONFIG['output_dim'], activation='selu')
        self.output_layer = Dense(1)

    def call(self, inputs, training=None):
        cgm_input, other_input = inputs
        
        # Procesamiento inicial
        x = self.flatten(cgm_input)
        x = Concatenate()([x, other_input])
        
        # Feature masking
        if training:
            feature_mask = self.feature_dropout(tf.ones_like(x))
            x = tf.multiply(x, feature_mask)
        
        # Pasos de decisión
        step_outputs = []
        entropy_loss = 0.0
        
        for transformer in self.transformers:
            step_output = transformer(x, training=training)
            
            # Feature selection
            attention_mask = Dense(x.shape[-1])(step_output)
            mask = custom_softmax(attention_mask)
            masked_x = tf.multiply(x, mask)
            
            step_outputs.append(masked_x)
            
            if training:
                # Calcular entropía
                entropy = tf.reduce_mean(tf.reduce_sum(
                    -mask * tf.math.log(mask + 1e-15), axis=1
                ))
                entropy_loss += entropy
        
        # Combinar salidas con atención
        combined = tf.stack(step_outputs, axis=1)
        attention_weights = Dense(len(step_outputs), activation='softmax')(
            tf.reduce_mean(combined, axis=2)
        )
        x = tf.reduce_sum(
            combined * tf.expand_dims(attention_weights, -1),
            axis=1
        )
        
        # Actualizar métrica de entropía
        if training:
            entropy_loss *= TABNET_CONFIG['sparsity_coefficient']
            self.entropy_tracker.update_state(entropy_loss)
            self.add_loss(entropy_loss)
        
        # Capas finales con residual
        x = self.final_dense1(x)
        x = self.final_norm1(x)
        x = self.final_dropout(x, training=training)
        
        skip = x
        x = self.final_dense2(x)
        x = self.final_norm2(x)
        x = self.final_dense3(x)
        x = tf.keras.layers.Add()([x, skip])
        
        return self.output_layer(x)

def create_tabnet_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo TabNet mejorado.
    
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
    model = TabNetModel(cgm_shape, other_features_shape)
    
    # Build model
    dummy_cgm = tf.keras.layers.Input(shape=cgm_shape[1:])
    dummy_other = tf.keras.layers.Input(shape=(other_features_shape[1],))
    model([dummy_cgm, dummy_other])
    
    return model