import numpy as np
from torch.utils.data import Dataset
from data import synthetic_data_loader


CACHE = {}

DSET_SPLIT_SIZES = {
    'train': [0, 0.8**2],
    'val': [0.8**2, 0.8],
    'test': [0.8, 1],
    'pred': [0, 1],
    None: [0, 1],
}


class SynthDataset(Dataset):
    '''
    Subtype of torch.utils.data.Dataset.
    for more info: http://pytorch.org/docs/master/data.html
    This class use the (copied) cope from the reference paper's official repo
    https://github.com/baumgach/vagan-code
    '''

    def __init__(self, opt, anomaly, mode='train', transform=None):
        super(SynthDataset, self).__init__()
        self.transform = transform
        if 'loaded' not in CACHE:
            self.load_cache(opt)
        split_size = DSET_SPLIT_SIZES[mode]
        idxs = np.where(CACHE['y'] == int(anomaly))[0]
        l1 = int(len(idxs) * split_size[0])
        l2 = int(len(idxs) * split_size[1])
        self.idxs = idxs[l1:l2]

    def load_cache(self, opt):
        data = synthetic_data_loader.load_and_maybe_generate_data(output_folder=opt.dataset_root,
                                                                  image_size=opt.image_size,
                                                                  force_overwrite=False)

        lhr_size = data['features'].shape[0]
        imsize = int(np.sqrt(lhr_size))

        images = np.reshape(data['features'][:], [imsize, imsize, -1])
        images = np.transpose(images, [2, 0, 1])

        masks = np.reshape(data['gt'][:], [imsize, imsize, -1])
        masks = np.transpose(masks, [2, 0, 1])

        labels = data['labels'][:]

        CACHE['X'] = images
        CACHE['y'] = labels
        CACHE['masks'] = masks
        CACHE['loaded'] = True

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        x = CACHE['X'][idx]
        y = CACHE['y'][idx]
        mask = CACHE['masks'][idx]

        x = np.expand_dims(x, 0)

        if self.transform:
            x = self.transform(x)
        return x, y, mask

if __name__ == '__main__':
    import torch
    dset = SynthDataset(None, True)
    healthy_dataloader = torch.utils.data.DataLoader(dset, batch_size=64,
                                                     shuffle=True, drop_last=True)
    for batch in healthy_dataloader:
        print(batch)
        break
