"""Model configuration parameters"""

###########################################################
###                Deep Learning Models                 ###
###########################################################
TCN_CONFIG = {
    'filters': [32, 64, 128],
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16],
    'dropout_rate': [0.2, 0.1],
    'activation': 'gelu',
    'epsilon': 1e-6,
    'use_layer_norm': True,
    'use_weight_norm': True,
    'use_spatial_dropout': True,
    'residual_dropout': 0.1
}

TRANSFORMER_CONFIG = {
    'num_heads': 8,
    'key_dim': 64,
    'num_layers': 4,
    'ff_dim': 256,
    'dropout_rate': 0.1,
    'epsilon': 1e-6,
    'activation': 'gelu',
    'use_relative_pos': True,
    'max_position': 32,
    'head_size': 32,
    'use_bias': True,
    'prenorm': True
}

WAVENET_CONFIG = {
    'filters': [32, 64, 128],
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16],
    'dropout_rate': 0.2,
    'use_gating': True,
    'use_skip_scale': True,
    'use_residual_scale': 0.1,
    'activation': 'elu'
}

TABNET_CONFIG = {
    'feature_dim': 128,
    'output_dim': 64,
    'num_decision_steps': 8,
    'relaxation_factor': 1.5,
    'sparsity_coefficient': 1e-4,
    'batch_momentum': 0.98,
    'virtual_batch_size': 128,
    'num_attention_heads': 4,
    'attention_dropout': 0.2,
    'feature_dropout': 0.1
}

ATTENTION_CONFIG = {
    'num_heads': 8,
    'key_dim': 64,
    'num_layers': 4,
    'ff_dim': 256,
    'dropout_rate': 0.1,
    'use_relative_attention': True,
    'max_relative_position': 32,
    'activation': 'gelu',
    'head_size': 32,
    'use_mask_future': False,
    'layer_dropout': 0.1
}

GRU_CONFIG = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.2,
    'epsilon': 1e-5,
    'attention_heads': 4
}

CNN_CONFIG = {
    'filters': [32, 64, 128, 256],
    'kernel_size': 3,
    'pool_size': 2,
    'dropout_rate': 0.2,
    'use_se_block': True,
    'se_ratio': 16,
    'use_layer_norm': True,
    'activation': 'gelu',
    'dilation_rates': [1, 2, 4]
}

GRU_CONFIG = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.2,
    'epsilon': 1e-5,
    'attention_heads': 4
}

RNN_CONFIG = {
    'hidden_units': [64, 32],
    'dropout_rate': 0.2,
    'recurrent_dropout': 0.1,
    'bidirectional': True,
    'epsilon': 1e-6,
    'use_time_distributed': True,
    'activation': 'relu'
}

LSTM_CONFIG = {
    'hidden_units': [64, 128, 256],
    'dense_units': [128, 64], 
    'dropout_rate': 0.25,
    'recurrent_dropout': 0.15,
    'epsilon': 1e-6,
    'attention_heads': 4,
    'use_bidirectional': True,
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'dense_activation': 'gelu', 
    'use_layer_norm': True
}