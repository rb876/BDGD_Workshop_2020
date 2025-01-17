import os
import torch
import odl
import time
import numpy as np
from torchvision import transforms

class GenEllipsesSamples:
    def __init__(self, space):
        self.space = space
    def _get_smpl(self):
        def _get_ellipse():
            return (
                ( np.abs(np.random.rand() - 0.5) * np.random.exponential(0.5) ),
                np.random.exponential() * 0.5,
                np.random.exponential() * 0.5,
                np.random.rand() - 0.5,
                np.random.rand() - 0.5,
                np.random.rand() * 2 * np.pi,
                )

        n = np.random.randint(5, 10)
        ellipses = [_get_ellipse() for _ in range(n)]
        phantom = odl.phantom.ellipsoid_phantom(self.space, ellipses).asarray()
        phantom = phantom - phantom.min()
        phantom = phantom / phantom.max()
        return phantom

class DatasetConstructor:
    def __init__(self, img_mode, train_size, dataset_name=None):
        self.img_mode = img_mode
        self.train_size = train_size
        self.gen_train_samples = GenEllipsesSamples(img_mode.space)
        self.dataset_name = dataset_name

    def data(self):
        if self._is_data():
            return self._load_data()
        else:
            start = time.time()
            data = self._dataset()
            print('============= time to generate dataset {:.4f} ============= \n'.format(time.time() - start), flush=True)
            return data

    def _dataset(self):
        trainY, trainX, train_initX = self._get_train(num_reps=self.train_size, threads=4)
        testY, testX, test_initX  = self._get_train(num_reps=100, threads=4, test=True)
        data = {'train': (trainY, trainX, train_initX),\
            'test': (testY, testX, test_initX), 'validation': (testY, testX, test_initX)}
        print('data generated -- training size {} \n'.format(self.train_size), flush=True)
        self._save_data(data)
        return data

    def _get_train(self, num_reps=4e3, threads=4, test=False):

        phantom = torch.stack([torch.from_numpy(self.gen_train_samples._get_smpl()) for _ in range(num_reps)])

        Y_, X_, initX_ = [], [], []
        from concurrent.futures import ThreadPoolExecutor
        def sinogram_wrapper(chunk):
            Y, X, initX = self.img_mode.sinogram(chunk)
            Y_.append(Y.unsqueeze(dim=1)),
            X_.append(X.unsqueeze(dim=1)),
            initX_.append(initX.unsqueeze(dim=1))

        batch_size = int(num_reps / threads)
        with ThreadPoolExecutor(max_workers = threads) as executor:
            executor.map( sinogram_wrapper, [chunk for chunk in torch.split(phantom, batch_size)] )

        if test:
            sl = torch.from_numpy(odl.phantom.shepp_logan(self.img_mode.space, modified=True).asarray()).unsqueeze(dim=0)
            Y, X, initX = self.img_mode.sinogram(sl)
            Y_.append(Y.unsqueeze(dim=1)),
            X_.append(X.unsqueeze(dim=1)),
            initX_.append(initX.unsqueeze(dim=1))

        return torch.cat(Y_), torch.cat(X_), torch.cat(initX_)

    def _save_data(self, data, path='./datasets/'):
        import pickle
        filename = path + self.dataset_name + '.p'
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    def _load_data(self, path='./datasets/'):
        import pickle
        filename = path + self.dataset_name + '.p'
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        handle.close()
        print('data loaded -- filepath: {} \n'.format(filename), flush=True)
        return data

    def _is_data(self, path='./datasets/'):
        import os
        filename = path + self.dataset_name + '.p'
        return os.path.isfile(filename)
