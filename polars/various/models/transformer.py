from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
from .config import TRANSFORMER_CONFIG

def create_transformer_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo Transformer con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo Transformer compilado
    """
    # Entrada CGM
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    
    # Transformer block
    attention = MultiHeadAttention(num_heads=TRANSFORMER_CONFIG['num_heads'], key_dim=TRANSFORMER_CONFIG['key_dim'])(cgm_input, cgm_input)
    attention = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(attention + cgm_input)
    
    # Feed-forward network
    ff = Dense(128, activation='relu')(attention)
    ff = Dense(cgm_shape[-1])(ff)
    ff = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(ff + attention)
    
    # Global pooling
    pooled = GlobalAveragePooling1D()(ff)
    
    # Entrada de otras características
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Combinar características
    combined = Concatenate()([pooled, other_input])
    
    # Capas densas finales
    dense = Dense(64, activation='relu')(combined)
    dense = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(dense)
    dense = Dropout(TRANSFORMER_CONFIG['dropout_rate'])(dense)
    
    output = Dense(1, activation='linear')(dense)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)