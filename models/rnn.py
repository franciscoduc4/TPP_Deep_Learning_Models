import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, SimpleRNN, Dropout, BatchNormalization,
    Concatenate, Bidirectional, TimeDistributed, MaxPooling1D
)
from keras.saving import register_keras_serializable
from .config import RNN_CONFIG

def create_rnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo RNN optimizado para velocidad con procesamiento temporal distribuido.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo RNN compilado
    """
    # Entradas
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Procesamiento temporal distribuido inicial
    if RNN_CONFIG['use_time_distributed']:
        x = TimeDistributed(Dense(32, activation=RNN_CONFIG['activation']))(cgm_input)
        x = TimeDistributed(BatchNormalization(epsilon=RNN_CONFIG['epsilon']))(x)
    else:
        x = cgm_input
    
    # Reducir secuencia temporal para procesamiento más rápido
    x = MaxPooling1D(pool_size=2)(x)
    
    # Capas RNN con menos unidades pero bidireccionales
    for units in RNN_CONFIG['hidden_units']:
        rnn_layer = SimpleRNN(
            units,
            activation=RNN_CONFIG['activation'],
            dropout=RNN_CONFIG['dropout_rate'],
            recurrent_dropout=RNN_CONFIG['recurrent_dropout'],
            return_sequences=True,
            unroll=True  # Desenrollar para secuencias cortas
        )
        
        if RNN_CONFIG['bidirectional']:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)
            
        x = BatchNormalization(
            epsilon=RNN_CONFIG['epsilon'],
            momentum=0.9  # Aumentar momentum para actualización más rápida
        )(x)
    
    # Último RNN sin return_sequences
    final_rnn = SimpleRNN(
        RNN_CONFIG['hidden_units'][-1],
        activation=RNN_CONFIG['activation'],
        dropout=RNN_CONFIG['dropout_rate'],
        recurrent_dropout=RNN_CONFIG['recurrent_dropout'],
        unroll=True
    )
    
    if RNN_CONFIG['bidirectional']:
        x = Bidirectional(final_rnn)(x)
    else:
        x = final_rnn(x)
    
    # Combinar características
    x = Concatenate()([x, other_input])
    
    # Reducir capas densas
    x = Dense(32, activation=RNN_CONFIG['activation'])(x)
    x = BatchNormalization(epsilon=RNN_CONFIG['epsilon'])(x)
    x = Dropout(RNN_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)