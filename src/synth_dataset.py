import cv2
import os
import numpy as np
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    '''
    Subtype of torch.utils.data.Dataset.
    for more info: http://pytorch.org/docs/master/data.html
    '''

    def __init__(self, image_size, root_dir='../dataset/', anomaly=False, transform=None):
        super(SynthDataset, self).__init__()
        self.root_dir = root_dir
        self.anomaly = anomaly
        self.transform = transform

        check_root = os.path.join(self.root_dir, 'data')
        self.names = [os.path.splitext(f)[0] for f in os.listdir(check_root) if os.path.isfile(os.path.join(check_root, f))]

        if anomaly is not None:
            self.names = [n for n in self.names if bool(int(n.split('_')[1])) == bool(anomaly)]

        x_paths = [os.path.join(self.root_dir, 'data', n) + '.png' for n in self.names]

        self.X = np.zeros((len(x_paths), image_size, image_size, 1))
        for i in range(len(x_paths)):
            self.X[i] = self.load_sample(x_paths[i])

        self.labels = np.asarray([int(n.split('_')[1]) for n in self.names])
        self.subtypes = np.asarray([int(n.split('_')[2]) for n in self.names])

        self.mean = self.X.mean(1)
        self.std = self.X.std()

    def load_sample(self, path):
        '''
        TODO test scipy.misc.imread(path)
        '''
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 2)
        return img

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx], self.subtypes[idx]


