# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
from sklearn.model_selection import train_test_split
from data import synthetic_data_loader
from data.batch_provider import BatchProvider


class synthetic_data():

    def __init__(self, exp_config):

        data = synthetic_data_loader.load_and_maybe_generate_data(output_folder=exp_config.preproc_folder,
                                                                  image_size=exp_config.image_size,
                                                                  effect_size=exp_config.effect_size,
                                                                  num_samples=exp_config.num_samples,
                                                                  moving_effect=exp_config.moving_effect,
                                                                  scale_to_one=exp_config.rescale_to_one,
                                                                  force_overwrite=False)

        self.data = data

        lhr_size = data['features'].shape[0]
        imsize = int(np.sqrt(lhr_size))

        images = np.reshape(data['features'][:], [imsize, imsize, -1])
        images = np.transpose(images, [2, 0, 1])

        gts = np.reshape(data['gt'][:], [imsize, imsize, -1])
        gts = np.transpose(gts, [2, 0, 1])

        labels = data['labels'][:]

        images_train_and_val, images_test, labels_train_and_val, labels_test, _, gts_test = train_test_split(
            images, labels, gts, test_size=0.2,  stratify=labels, random_state=42)
        images_train, images_val, labels_train, labels_val = train_test_split(
            images_train_and_val, labels_train_and_val, test_size=0.2, stratify=labels_train_and_val, random_state=42)

        self.images_test = images_test
        self.labels_test = labels_test
        self.gts_test = gts_test

        N_train = images_train.shape[0]
        N_test = images_test.shape[0]
        N_val = images_val.shape[0]

        train_indices = np.arange(N_train)
        train_AD_indices = train_indices[np.where(labels_train == 1)]
        train_CN_indices = train_indices[np.where(labels_train == 0)]

        test_indices = np.arange(N_test)
        test_AD_indices = test_indices[np.where(labels_test == 1)]
        test_CN_indices = test_indices[np.where(labels_test == 0)]

        val_indices = np.arange(N_val)
        val_AD_indices = val_indices[np.where(labels_val == 1)]
        val_CN_indices = val_indices[np.where(labels_val == 0)]

        # Create the batch providers
        self.trainAD = BatchProvider(
            images_train, labels_train, train_AD_indices)
        self.trainCN = BatchProvider(
            images_train, labels_train, train_CN_indices)

        self.validationAD = BatchProvider(
            images_val, labels_val, val_AD_indices)
        self.validationCN = BatchProvider(
            images_val, labels_val, val_CN_indices)

        self.testAD = BatchProvider(images_test, labels_test, test_AD_indices)
        self.testCN = BatchProvider(images_test, labels_test, test_CN_indices)

        self.train = BatchProvider(images_train, labels_train, train_indices)
        self.validation = BatchProvider(images_val, labels_val, val_indices)
        self.test = BatchProvider(images_test, labels_test, test_indices)


if __name__ == '__main__':

    # If called as main perform debugging

    import matplotlib.pyplot as plt
    from classifier.experiments import synthetic_normalnet as exp_config

    data = synthetic_data(exp_config)

    print('DEBUGGING OUTPUT')
    print('training')
    for ii in range(2):
        X_tr, Y_tr = data.trainCN.next_batch(10)
        print(np.mean(X_tr))
        print(np.max(X_tr))
        print(np.min(X_tr))
        print(X_tr.shape)
        print(Y_tr)
        print('--')

    print('test')
    for ii in range(2):
        X_te, Y_te = data.trainAD.next_batch(10)
        print(np.mean(X_te))
        print(X_te.shape)
        print(Y_te)
        print('--')

    print('validation')
    for ii in range(2):
        X_va, Y_va = data.validationAD.next_batch(10)
        print(np.mean(X_va))
        print(X_va.shape)
        print(Y_va)
        print('--')

    # import matplotlib.pyplot as plt
    print('validation')
    for [X_va, Y_va] in data.train.iterate_batches(10):
        print(np.mean(X_va))
        print(X_va.shape)
        print(Y_va)

        plt.imshow(np.squeeze(X_va[0, ...]), cmap='gray')
        plt.show()

        print('--')
