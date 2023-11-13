
config = {

    # Satellite configuration
    'satellite': 'PRISMA',
    'ratio': 6,
    'nbits': 16,

    # Training settings
    'save_weights': True,
    'save_weights_path': 'weights',
    'save_training_stats': False,

    # Training hyperparameters
    'learning_rate': 0.00005,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epochs': 1000,
    'semi_width': 18,

    'alpha_1': 0.5,
    'alpha_2': 0.25,

    'last_wl': 770,

    'num_blocks': 2,    # Number of hyperspectral blocks
    'n_components': 4,  # Number of principal components to extract

}