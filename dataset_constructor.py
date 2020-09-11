import os
import torch
import odl
import time
import numpy as np
import foam_ct_phantom
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

class GenFoamSamples:
    def __init__(self, space):
        self.space = space

    def _get_smpl(self, num_reps, test, path='./datasets/foam_phantoms'):
        flag = 'train' if test == False else 'test'
        path = os.path.join(path, flag)

        def _gen_phantoms():
            if not os.path.isdir(path):
                os.makedirs(path)
            if not os.listdir(path):
                print('foam_phantoms is empty')
                for idx in range(num_reps):
                    filename_phantom = os.path.join(path, 'phantom_' + str(idx) + '.h5')
                    foam_ct_phantom.FoamPhantom.generate(filename_phantom, 12345, nspheres_per_unit=1000)
            else:
                print('foam phantoms have already been created')

        _gen_phantoms()
        X_ = []
        geom = foam_ct_phantom.VolumeGeometry(self.space.shape[0], self.space.shape[1], 1, 3/256)
        for idx in range(num_reps):
            filename_phantom = os.path.join(path, 'phantom_' + str(idx) + '.h5')
            phantom = foam_ct_phantom.FoamPhantom(filename_phantom)
            filename_volume = os.path.join(path, 'midslice_' + str(idx) + '.h5')
            phantom.generate_volume(filename_volume, geom)
            X_.append( torch.from_numpy( foam_ct_phantom.load_volume(filename_volume).squeeze() ) )

        if not test:
            # data augmentation
            aug_fct = 4
            f = lambda x: transforms.Compose([  transforms.ToPILImage(), \
                                                transforms.RandomHorizontalFlip(), \
                                                transforms.RandomVerticalFlip(), \
                                                transforms.RandomResizedCrop( (self.space.shape[0], self.space.shape[1]) ), \
                                                transforms.ToTensor()
                                                ])(x)
            X = torch.stack(X_)
            N, H, W = X.shape[-3], X.shape[-2], X.shape[-1]
            X = X.unsqueeze(dim=0).expand(aug_fct, N, H, W).reshape(aug_fct*N, H, W)
            X = torch.stack([f(tensor).squeeze() for tensor in X])
        else:
            X = torch.stack(X_)

        return X

class DatasetConstructor:
    def __init__(self, img_mode, train_size, dataset_name=None, dataset_type='GenFoamSamples'):
        self.img_mode = img_mode
        self.train_size = train_size
        if dataset_type == GenEllipsesSamples.__name__:
            self.gen_train_samples = GenEllipsesSamples(img_mode.space)
        elif dataset_type == GenFoamSamples.__name__:
            self.gen_train_samples = GenFoamSamples(img_mode.space)
        else:
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

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
        testY, testX, test_initX  = self._get_train(num_reps=10, threads=4, test=True)
        data = {'train': (trainY, trainX, train_initX),\
            'test': (testY, testX, test_initX), 'validation': (testY, testX, test_initX)}
        print('data generated -- training size {} \n'.format(self.train_size), flush=True)
        self._save_data(data)
        return data

    def _get_train(self, num_reps=4e3, threads=4, test=False):

        if self.dataset_type == GenEllipsesSamples.__name__:
            phantom = torch.stack([torch.from_numpy(self.gen_train_samples._get_smpl()) for _ in range(num_reps)])
        elif self.dataset_type == GenFoamSamples.__name__:
            phantom = self.gen_train_samples._get_smpl(num_reps, test)
        else:
            NotImplementedError

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

        if test and \
            self.dataset_type == GenEllipsesSamples.__name__:
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
