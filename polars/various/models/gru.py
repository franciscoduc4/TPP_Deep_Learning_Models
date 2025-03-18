from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, BatchNormalization, Concatenate
)
from .config import GRU_CONFIG

def create_gru_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    '''
    Crea un modelo GRU (Gated Recurrent Unit) con entrada dual para datos CGM y otras características.

    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)

    Retorna:
    --------
    Model
        Modelo GRU compilado
    '''
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    x = GRU(GRU_CONFIG['hidden_units'][0], return_sequences=True)(cgm_input)
    x = GRU(GRU_CONFIG['hidden_units'][1])(x)
    x = BatchNormalization()(x)
    
    x = Concatenate()([x, other_input])
    x = Dense(GRU_CONFIG['hidden_units'][1], activation='relu')(x)
    x = Dropout(GRU_CONFIG['dropout_rate'])(x)
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)