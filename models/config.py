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

###########################################################
###         Deep Reinforcement Learning Models          ###
###########################################################

PPO_CONFIG = {
    'hidden_units': [256, 256, 128, 64],
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'dropout_rate': 0.1,
    'epsilon': 1e-5
}

DQN_CONFIG = {
    'hidden_units': [256, 256, 128, 64],
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_capacity': 100000,
    'batch_size': 64,
    'target_update_freq': 1000,
    'activation': 'relu',
    'dropout_rate': 0.1,
    'epsilon': 1e-5,
    'dueling': True,
    'double': True,
    'prioritized': False,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'priority_beta_increment': 1e-3
}

A2C_A3C_CONFIG = {
    'hidden_units': [256, 256, 128, 64],
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'lambda': 0.95,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'dropout_rate': 0.1,
    'epsilon': 1e-5
}

DDPG_CONFIG = {
    # Architecture
    'actor_hidden_units': [400, 300],
    'critic_hidden_units': [400, 300],
    'actor_activation': 'relu',
    'critic_activation': 'relu',
    # Optimization
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
    'tau': 0.001,
    # Replay buffer
    'buffer_capacity': 100000,
    'batch_size': 64,
    # Exploration
    'noise_std': 0.2,
    # Regularization
    'dropout_rate': 0.0,
    'epsilon': 1e-6
}

SAC_CONFIG = {
    # Architecture
    'actor_hidden_units': [256, 256],
    'critic_hidden_units': [256, 256],
    'actor_activation': 'relu',
    'critic_activation': 'relu',
    # Optimization
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'alpha_lr': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,
    # Replay buffer
    'buffer_capacity': 1000000,
    'batch_size': 256,
    # Entropy regularization
    'initial_alpha': 0.2,
    'log_std_min': -20,
    'log_std_max': 2,
    # Regularization
    'dropout_rate': 0.0,
    'epsilon': 1e-6
}

TRPO_CONFIG = {
    # Architecture
    'hidden_units': [64, 64],          # Unidades en capas ocultas
    'use_layer_norm': True,            # Usar layer normalization
    'epsilon': 1e-6,                   # Epsilon para normalización y estabilidad numérica
    
    # TRPO specific
    'delta': 0.01,
    'gamma': 0.99,
    'lambda': 0.95,
    'critic_learning_rate': 3e-4,
    'backtrack_iters': 10,
    'backtrack_coeff': 0.8,
    'cg_iters': 10,
    'damping': 0.1,
    # Training
    'iterations': 500,
    'min_steps_per_update': 2048,
    'value_epochs': 10,
    'batch_size': 64,
    'evaluate_interval': 10
}

###########################################################
###           Reinforcement Learning Models             ###
###########################################################