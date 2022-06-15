from dldemos.DeepNetwork.dataset import get_cat_set
from dldemos.DeepNetwork.model import DeepNetwork, train


def main():
    train_X, train_Y, test_X, test_Y = get_cat_set(
        'dldemos/LogisticRegression/data/archive/dataset', train_size=1500)
    n_x = train_X.shape[0]
    model = DeepNetwork([n_x, 30, 30, 20, 20, 1],
                        ['relu', 'relu', 'relu', 'relu', 'sigmoid'])
    train(model,
          train_X,
          train_Y,
          1,
          learning_rate=0.01,
          print_interval=10,
          test_X=test_X,
          test_Y=test_Y)


if __name__ == '__main__':
    main()
