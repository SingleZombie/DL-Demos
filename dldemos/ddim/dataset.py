import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import os

CELEBA_HQ_DIR = 'data/celebA/celeba_hq_256'


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
    tensor = transforms.ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


class MNISTImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.mnist = torchvision.datasets.MNIST(root='./data/mnist')

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index: int):
        img = self.mnist[index][0]
        pipeline = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Lambda(lambda x: (x - 0.5) * 2)])
        return pipeline(img)


class CelebADataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: (x - 0.5) * 2)])
        return pipeline(img)


def get_dataloader(type,
                   batch_size,
                   dist_train=False,
                   num_workers=4):
    if type == 'CelebAHQ':
        dataset = CelebADataset(CELEBA_HQ_DIR)
    elif type == 'MNIST':
        dataset = MNISTImageDataset()
    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        return dataloader


def get_img_shape(dataset_type):
    if dataset_type == 'MNIST':
        return (1, 28, 28)
    elif dataset_type == 'CelebAHQ':
        return (3, 256, 256)


if __name__ == '__main__':
    import os
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
