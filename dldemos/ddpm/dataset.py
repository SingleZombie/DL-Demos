import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader


def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp.jpg')
    tensor = ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_img_shape():
    return (1, 28, 28)


if __name__ == '__main__':
    download_dataset()