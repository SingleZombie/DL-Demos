import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp.jpg')
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size: int):
    dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                         transform=ToTensor())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_img_shape():
    return (1, 28, 28)


if __name__ == '__main__':
    import os
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
