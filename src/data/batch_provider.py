# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np


class BatchProvider():
    """
    This is a helper class to conveniently access mini batches of training, testing and validation data
    """

    # indices don't always cover all of X and Y (e.g. in the case of val set)
    def __init__(self, X, y, indices):

        self.X = X
        self.y = y
        self.indices = indices
        self.unused_indices = indices.copy()

    def next_batch(self, batch_size, add_dummy_dimension=True):
        """
        Get a single random batch. This implements sampling without replacement (not just on a batch level), this means
        all the data gets sampled eventually.
        """

        if len(self.unused_indices) < batch_size:
            self.unused_indices = self.indices

        batch_indices = np.random.choice(
            self.unused_indices, batch_size, replace=False)
        self.unused_indices = np.setdiff1d(self.unused_indices, batch_indices)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        X_batch = self.X[batch_indices, ...]
        y_batch = self.y[batch_indices, ...]

        if add_dummy_dimension:
            X_batch = np.expand_dims(X_batch, axis=-1)

        return X_batch, y_batch

    def iterate_batches(self, batch_size, add_dummy_dimension=True):
        """
        Get a range of batches. Use as argument of a for loop like you would normally use
        the range() function.
        """

        np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            # if b_i + batch_size > N:
            #     continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])

            X_batch = self.X[batch_indices, ...]
            y_batch = self.y[batch_indices, ...]

            if add_dummy_dimension:
                X_batch = np.expand_dims(X_batch, axis=-1)

            yield X_batch, y_batch
