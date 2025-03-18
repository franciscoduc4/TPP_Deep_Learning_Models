"""Model configuration parameters"""

TCN_CONFIG = {
    'filters': 64,
    'kernel_size': 3,
    'dilations': [2**i for i in range(4)],
    'dropout_rate': [0.3, 0.2],
    'epsilon': 1e-6
}

TRANSFORMER_CONFIG = {
    'num_heads': 4,
    'key_dim': 32,
    'ff_dim': 128,
    'dropout_rate': 0.2,
    'epsilon': 1e-6
}

WAVENET_CONFIG = {
    'filters': [32, 64, 128],
    'kernel_size': 2,
    'dilations': [2**i for i in range(8)],  # [1, 2, 4, 8, 16, 32, 64, 128]
    'dropout_rate': 0.2
}

TABNET_CONFIG = {
    'feature_dim': 64,
    'output_dim': 32,
    'num_decision_steps': 5,
    'relaxation_factor': 1.5,
    'sparsity_coefficient': 1e-5,
    'batch_momentum': 0.98
}

ATTENTION_CONFIG = {
    'num_heads': 8,
    'key_dim': 64,
    'num_layers': 4,
    'ff_dim': 256,
    'dropout_rate': 0.1
}

GRU_CONFIG = {
    'hidden_units': [128, 64],
    'dropout_rate': 0.2
}

CNN_CONFIG = {
    'filters': [32, 64, 128, 256],
    'kernel_size': 3,
    'pool_size': 2,
    'dropout_rate': 0.2
}

RNN_CONFIG = {
    'hidden_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.2,
    'bidirectional': True,
    'epsilon': 1e-6
}