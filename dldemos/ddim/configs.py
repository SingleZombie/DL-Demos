mnist_cfg = {
    'dataset_type': 'MNIST',
    'img_shape': [1, 28, 28],
    'model_path': 'dldemos/ddim/mnist.pth',
    'batch_size': 512,
    'n_epochs': 50,
    'channels': [10, 20, 40, 80],
    'pe_dim': 128
}

# Deprecated config. It's for model in `network_my.py`
celebahq_cfg1 = {
    'dataset_type': 'CelebAHQ',
    'img_shape': [3, 128, 128],
    'model_path': 'dldemos/ddim/celebahq1.pth',
    'batch_size': 64,
    'n_epochs': 1000,
    'channels': [64, 128, 256, 512, 512],
    'pe_dim': 128,
    'with_attn': [False, False, False, True, False]
}
celebahq_cfg2 = {
    'dataset_type': 'CelebAHQ',
    'img_shape': [3, 64, 64],
    'model_path': 'dldemos/ddim/celebahq2.pth',
    'batch_size': 128,
    'n_epochs': 2500,
    'scheduler_cfg': {
        'lr': 5e-4,
        'milestones': [1500, 2100],
        'gamma': 0.1,
    },
    'channels': [128, 256, 512, 512],
    'pe_dim': 128,
    'with_attn': [False, False, True, True],
    'norm_type': 'gn'
}
celebahq_cfg3 = {
    'dataset_type': 'CelebAHQ',
    'img_shape': [3, 128, 128],
    'model_path': 'dldemos/ddim/celebahq3.pth',
    'batch_size': 32,
    'n_epochs': 1500,
    'scheduler_cfg': {
        'lr': 2e-4,
        'milestones': [800, 1300],
        'gamma': 0.1,
    },
    'channels': [128, 256, 256, 512, 512],
    'pe_dim': 128,
    'with_attn': [False, False, False, True, True],
    'norm_type': 'gn'
}
celebahq_cfg4 = {
    'dataset_type': 'CelebAHQ',
    'img_shape': [3, 256, 256],
    'model_path': 'dldemos/ddim/celebahq4.pth',
    'batch_size': 8,
    'n_epochs': 1000,
    'scheduler_cfg': {
        'lr': 2e-5,
        'milestones': [800],
        'gamma': 0.1,
    },
    'channels': [128, 128, 256, 256, 512, 512],
    'pe_dim': 128,
    'with_attn': [False, False, False, False, True, True],
    'norm_type': 'gn'
}

configs = [
    mnist_cfg, celebahq_cfg1, celebahq_cfg2, celebahq_cfg3, celebahq_cfg4
]
