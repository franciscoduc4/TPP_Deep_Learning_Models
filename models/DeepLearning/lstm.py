import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, Concatenate,
    MultiHeadAttention, Add, GlobalAveragePooling1D, GlobalMaxPooling1D,
    BatchNormalization, Bidirectional
)
from keras.saving import register_keras_serializable
from ..config import LSTM_CONFIG

def create_lstm_attention_block(x, units, num_heads=4, dropout_rate=0.2) -> tf.Tensor:
    """
    Crea un bloque LSTM con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    x : tensor
        Tensor de entrada
    units : int
        Número de unidades LSTM
    num_heads : int
        Número de cabezas de atención
    dropout_rate : float
        Tasa de dropout
        
    Retorna:
    --------
    tensor
        Tensor procesado
    """
    # LSTM con skip connection
    skip1 = x
    x = LSTM(
        units,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=LSTM_CONFIG['recurrent_dropout'],
        activation=LSTM_CONFIG['activation'],
        recurrent_activation=LSTM_CONFIG['recurrent_activation']
    )(x)
    
    x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(x)
    if skip1.shape[-1] == units:
        x = Add()([x, skip1])
    
    # Multi-head attention con gating mechanism
    skip2 = x
    
    # Atención con proyección de valores
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=units // num_heads,
        value_dim=units // num_heads,
        dropout=dropout_rate
    )(x, x)
    
    # Mecanismo de gating para controlar flujo de información
    gate = Dense(units, activation='sigmoid')(skip2)
    attention_output = attention_output * gate
    
    # Conexión residual con normalización
    x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(attention_output + skip2)
    
    return x

def create_lstm_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo LSTM avanzado con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo LSTM compilado
    """
    # Entradas
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Proyección inicial
    x = Dense(LSTM_CONFIG['hidden_units'][0])(cgm_input)
    x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(x)
    
    # Bloques LSTM apilados con distinto nivel de complejidad
    for i, units in enumerate(LSTM_CONFIG['hidden_units']):
        # Opción de bidireccional para primeras capas si está configurado
        if i < len(LSTM_CONFIG['hidden_units'])-1 and LSTM_CONFIG['use_bidirectional']:
            lstm_layer = Bidirectional(LSTM(
                units,
                return_sequences=True,
                dropout=LSTM_CONFIG['dropout_rate'],
                recurrent_dropout=LSTM_CONFIG['recurrent_dropout'],
                activation=LSTM_CONFIG['activation'],
                recurrent_activation=LSTM_CONFIG['recurrent_activation']
            ))
            x = lstm_layer(x)
            x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(x)
        else:
            # Bloques con atención para capas posteriores
            x = create_lstm_attention_block(
                x, 
                units=units, 
                num_heads=LSTM_CONFIG['attention_heads'],
                dropout_rate=LSTM_CONFIG['dropout_rate']
            )
    
    # Extracción de características con pooling estadístico
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # Red densa final con skip connections
    skip = x
    x = Dense(LSTM_CONFIG['dense_units'][0], activation=LSTM_CONFIG['dense_activation'])(x)
    x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'])(x)
    
    # Segunda capa densa con residual
    x = Dense(LSTM_CONFIG['dense_units'][1], activation=LSTM_CONFIG['dense_activation'])(x)
    if skip.shape[-1] == LSTM_CONFIG['dense_units'][1]:
        x = Add()([x, skip])  # Skip connection si las dimensiones coinciden
        
    x = LayerNormalization(epsilon=LSTM_CONFIG['epsilon'])(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'] * 0.5)(x)  # Menor dropout en capas finales
    
    # Capa de salida
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)