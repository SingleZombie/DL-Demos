# Modify from
# https://github.com/xunge/pytorch_lmdb_imagenet/blob/master/folder2lmdb.py

import os
import os.path as osp
import pickle

import lmdb
import six
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


class MyImageFolder(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        return raw_reader(path)


def folder2lmdb(img_dir, output_path, write_frequency=5000):
    directory = img_dir
    print('Loading dataset from %s' % directory)
    dataset = MyImageFolder(directory)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = output_path
    isdir = os.path.isdir(lmdb_path)

    print('Generate LMDB to %s' % lmdb_path)
    db = lmdb.open(lmdb_path,
                   subdir=isdir,
                   map_size=1099511627776 * 2,
                   readonly=False,
                   meminit=False,
                   map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data(image))
        if idx % write_frequency == 0:
            print('[%d/%d]' % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print('Flushing database ...')
    db.sync()
    db.close()


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(Dataset):

    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path,
                             subdir=osp.isdir(db_path),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
