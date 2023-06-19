import torchvision

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import einops


def download_mnist():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp_mnist.jpg')
    tensor = transforms.ToTensor()(img)
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


class MNISTImageDataset(Dataset):

    def __init__(self, img_shape=(28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.mnist = torchvision.datasets.MNIST(root='./data/mnist')

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index: int):
        img = self.mnist[index][0]
        pipeline = transforms.Compose(
            [transforms.Resize(self.img_shape),
             transforms.ToTensor()])
        return pipeline(img)


def get_dataloader(type,
                   batch_size,
                   root='data/celebA/img_align_celeba',
                   img_shape=None,
                   dist_train=False,
                   **kwargs):
    if type == 'CelebA':
        if img_shape is not None:
            kwargs['img_shape'] = img_shape
        dataset = CelebADataset(root, **kwargs)
    elif type == 'MNIST':
        if img_shape is not None:
            dataset = MNISTImageDataset(img_shape)
        else:
            dataset = MNISTImageDataset()
    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    if os.path.exists('data/celebA/img_align_celeba'):
        dataloader = get_dataloader('CelebA', 16)
        img = next(iter(dataloader))
        print(img.shape)
        N = img.shape[0]
        img = einops.rearrange(img,
                               '(n1 n2) c h w -> c (n1 h) (n2 w)',
                               n1=int(N**0.5))
        img = transforms.ToPILImage()(img)
        img.save('work_dirs/tmp_celeba.jpg')

    download_mnist()
