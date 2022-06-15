import matplotlib.pyplot as plt
import numpy as np

LEN = 10

result_dict = {
    'batch_size_8': [
        0.6954, 0.6527, 0.5950, 0.5475, 0.4941, 0.6317, 0.4309, 0.4870, 0.4461,
        0.2928
    ],
    'batch_size_64': [
        0.6910, 0.6447, 0.6133, 0.5756, 0.5590, 0.5224, 0.5478, 0.4379, 0.4241,
        0.3764
    ],
    'batch_size_128': [
        0.6910, 0.6497, 0.6289, 0.6168, 0.5802, 0.5677, 0.5366, 0.5436, 0.5282,
        0.5344
    ],
    'batch_size_2000': [
        0.6966, 0.6840, 0.6770, 0.6780, 0.6675, 0.6572, 0.6605, 0.6482, 0.6719,
        0.6392
    ],
    'Momentum_64': [
        0.6917, 0.6581, 0.6212, 0.5774, 0.5123, 0.4700, 0.4162, 0.3581, 0.3168,
        0.2996
    ],
    'RMSProp_64': [
        0.6924, 0.6519, 0.6381, 0.6209, 0.6043, 0.5895, 0.5747, 0.5635, 0.5491,
        0.5363
    ],
    'Adam_64': [
        0.6781, 0.6150, 0.5801, 0.5466, 0.5163, 0.4881, 0.4617, 0.4365, 0.4154,
        0.3959
    ],
    'Adam_64_decay_0.2': [
        0.6861, 0.6021, 0.5783, 0.5644, 0.5544, 0.5471, 0.5409, 0.5357, 0.5314,
        0.5276
    ],
    'Adam_64_decay_0.005': [
        0.6900, 0.6047, 0.5558, 0.5283, 0.5068, 0.4843, 0.4462, 0.4307, 0.4145,
        0.3974
    ]
}


def plot_curves(result_keys):
    x = np.linspace(0, 90, LEN)
    for k in result_keys:
        y = result_dict[k]
        plt.plot(x, y, label=k)
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.legend()

    plt.show()


plot_curves(
    ['batch_size_8', 'batch_size_64', 'batch_size_128', 'batch_size_2000'])
plot_curves(['batch_size_64', 'Momentum_64', 'RMSProp_64', 'Adam_64'])
plot_curves(['Adam_64', 'Adam_64_decay_0.2', 'Adam_64_decay_0.005'])
