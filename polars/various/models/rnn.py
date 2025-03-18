import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, SimpleRNN, Dropout, BatchNormalization,
    Concatenate, Bidirectional
)
from .config import RNN_CONFIG

def create_rnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo RNN (Recurrent Neural Network) con capas bidireccionales y skip connections.
    
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
    
    # Capas RNN
    x = cgm_input
    skip_connections = []
    
    for units in RNN_CONFIG['hidden_units']:
        rnn_layer = SimpleRNN(
            units,
            dropout=RNN_CONFIG['dropout_rate'],
            recurrent_dropout=RNN_CONFIG['recurrent_dropout'],
            return_sequences=True
        )
        
        if RNN_CONFIG['bidirectional']:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)
            
        x = BatchNormalization(epsilon=RNN_CONFIG['epsilon'])(x)
        skip_connections.append(x)
    
    # Último RNN sin return_sequences
    final_rnn = SimpleRNN(
        RNN_CONFIG['hidden_units'][-1],
        dropout=RNN_CONFIG['dropout_rate'],
        recurrent_dropout=RNN_CONFIG['recurrent_dropout']
    )
    
    if RNN_CONFIG['bidirectional']:
        x = Bidirectional(final_rnn)(x)
    else:
        x = final_rnn(x)
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # Capas densas finales
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization(epsilon=RNN_CONFIG['epsilon'])(x)
    x = Dropout(RNN_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)