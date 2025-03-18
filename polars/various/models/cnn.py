from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    MaxPooling1D, GlobalAveragePooling1D, Concatenate
)
from .config import CNN_CONFIG

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
    
    # Capas CNN
    conv = Conv1D(filters=CNN_CONFIG['filters'][1], kernel_size=CNN_CONFIG['kernel_size'], activation='relu')(cgm_input)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    
    conv = Conv1D(filters=CNN_CONFIG['filters'][0], kernel_size=CNN_CONFIG['kernel_size'], activation='relu')(conv)
    conv = BatchNormalization()(conv)
    conv = GlobalAveragePooling1D()(conv)
    
    # Entrada de otras características
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Combinar características
    combined = Concatenate()([conv, other_input])
    
    # Capas densas
    dense = Dense(64, activation='relu')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'])(dense)
    
    output = Dense(1, activation='linear')(dense)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)
