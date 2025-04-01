from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, LayerNormalization, Concatenate,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from keras.saving import register_keras_serializable
from ..config import GRU_CONFIG

def create_gru_attention_block(x, units, num_heads=4):
    """
    Crea un bloque GRU con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    x : tensor
        Tensor de entrada
    units : int
        Número de unidades GRU
    num_heads : int
        Número de cabezas de atención
    """
    # GRU con skip connection
    skip1 = x
    x = GRU(
        units,
        return_sequences=True,
        dropout=GRU_CONFIG['dropout_rate'],
        recurrent_dropout=GRU_CONFIG['recurrent_dropout']
    )(x)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
    if skip1.shape[-1] == units:
        x = Add()([x, skip1])
    
    # Multi-head attention
    skip2 = x
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=units // num_heads
    )(x, x)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(attention_output + skip2)
    
    return x

def create_gru_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo GRU avanzado con self-attention y conexiones residuales.
    
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
    """
    # Entradas
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Proyección inicial
    x = Dense(GRU_CONFIG['hidden_units'][0])(cgm_input)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Bloques GRU con attention
    for units in GRU_CONFIG['hidden_units']:
        x = create_gru_attention_block(x, units)
    
    # Pooling global
    x = GlobalAveragePooling1D()(x)
    
    # Combinar con otras características
    combined = Concatenate()([x, other_input])
    
    # Red densa final con skip connections
    for units in [128, 64]:
        skip = combined
        x = Dense(units, activation='relu')(combined)
        x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
        x = Dropout(GRU_CONFIG['dropout_rate'])(x)
        if skip.shape[-1] == units:
            combined = Add()([x, skip])
        else:
            combined = x
    
    output = Dense(1)(combined)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)