from dldemos.AdvancedOptimizer.model import DeepNetwork, train
from dldemos.AdvancedOptimizer.optimizer import (Adam, GradientDescent,
                                                 Momentum, RMSProp,
                                                 get_hyperbola_func)
from dldemos.DeepNetwork.dataset import get_cat_set


def main():
    train_X, train_Y, dev_X, dev_Y = get_cat_set(
        'dldemos/LogisticRegression/data/archive/dataset', train_size=1000)
    n_x = train_X.shape[0]

    # train_X: [224*224*3, 2000]
    model = DeepNetwork([n_x, 30, 20, 20, 1],
                        ['relu', 'relu', 'relu', 'sigmoid'])

    # Please close the unused optimizers by comment marks

    optimizer = GradientDescent(model.save(), learning_rate=0.001)
    optimizer = Momentum(model.save(), learning_rate=0.001, from_scratch=True)
    optimizer = RMSProp(model.save(), learning_rate=0.00001, from_scratch=True)
    optimizer = Adam(model.save(), learning_rate=0.00001, from_scratch=True)

    lr_scheduler_1 = get_hyperbola_func(0.2)
    lr_scheduler_2 = get_hyperbola_func(0.005)

    optimizer = Adam(model.save(),
                     learning_rate=0.00001,
                     from_scratch=True,
                     lr_scheduler=lr_scheduler_1)

    optimizer = Adam(model.save(),
                     learning_rate=0.00001,
                     from_scratch=True,
                     lr_scheduler=lr_scheduler_2)

    train(model,
          optimizer,
          train_X,
          train_Y,
          100,
          model_name='model_64',
          save_dir='work_dirs',
          recover_from=None,
          batch_size=64,
          print_interval=10,
          dev_X=dev_X,
          dev_Y=dev_Y,
          plot_mini_batch=False)


if __name__ == '__main__':
    main()
