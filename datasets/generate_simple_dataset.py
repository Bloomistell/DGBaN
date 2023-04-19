import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler as mmScaler



class ring_dataset():
    def __init__(self, N=32, inner_scale=0.1, outer_scale=0.45, ring_resolution=1, train_fraction=0.8):
        self.N = N
        self.train_fraction = train_fraction

        self.centers = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)
        inners = []
        outers = []
        for inner in np.arange(N * inner_scale, N * outer_scale - N * 0.05, ring_resolution):
            for outer in np.arange(inner + N * 0.05, N * outer_scale, ring_resolution):
                inners.append(inner)
                outers.append(outer)

        self.inners = np.array(inners)
        self.outers = np.array(outers)

        self.img_coor = np.array([[(i, j) for i in range(self.N)] for j in range(self.N)], dtype=np.int32)
        self.n_features = 4

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu'):
        np.random.seed(seed)

        centers_idx = np.random.choice(self.N * self.N, size=data_size)
        radius = np.linalg.norm(self.img_coor[np.newaxis, :, :, :] - self.centers[centers_idx, np.newaxis, np.newaxis, :], axis=3)

        width_idx = np.random.choice(len(self.inners), size=data_size)
        self.img = np.zeros_like(radius, dtype=np.bool)
        self.img[np.logical_and(radius >= self.inners[width_idx, np.newaxis, np.newaxis], radius < self.outers[width_idx, np.newaxis, np.newaxis])] = 1

        features = np.concatenate([
            self.centers[centers_idx, 0].reshape(-1, 1),
            self.centers[centers_idx, 1].reshape(-1, 1),
            self.inners[width_idx].reshape(-1, 1),
            self.outers[width_idx].reshape(-1, 1)
        ], axis=1)

        self.scaler = mmScaler()
        self.features = self.scaler.fit_transform(features)

        train_dataset = TensorDataset(
            torch.tensor(self.features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            torch.tensor(self.img[:int(self.train_fraction * data_size)], dtype=torch.float, device=device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            torch.tensor(self.img[int(self.train_fraction * data_size):], dtype=torch.float, device=device)
        )
        
        return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def random_sample(self):
        center = self.centers[np.random.choice(self.N * self.N)]
        radius = np.linalg.norm(self.img_coor - center, axis=2)

        width_idx = np.random.choice(len(self.inners))
        img = np.zeros_like(radius, dtype=np.bool)
        img[np.logical_and(radius >= self.inners[width_idx], radius < self.outers[width_idx])] = 1

        features = np.array([center[0], center[1], self.inners[width_idx], self.outers[width_idx]]).reshape(1, -1)
        features = torch.tensor(self.scaler.transform(features), dtype=torch.float)

        return features, img



class randomized_ring_dataset():
    def __init__(self, N=32, train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.train_fraction = train_fraction

        self.centers = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)
        self.means = np.arange(N * 0.2, N * 0.4, N / 32)
        self.sigs = np.arange(N * 0.05, N * 0.2, N / 32)
        self.nrgs = np.arange(0.05 * self.N2, 0.21 * self.N2, 0.02 * self.N2).astype(np.int32)

        self.img_coor = np.array([(i, j) for i in range(self.N) for j in range(self.N)], dtype=np.int32)
        self.n_features = 5

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu', test_return=False):
        features_path = f'../save_dataset/features_N-{self.N}_data_size-{data_size}_seed-{seed}.npy'
        imgs_path = f'../save_dataset/images_N-{self.N}_data_size-{data_size}_seed-{seed}.npy'
        if os.path.exists(features_path):
            features = np.load(features_path)
            self.imgs = np.load(imgs_path)

        else:
            np.random.seed(seed)

            prob_distr, center, mean, sig = self.gaussian_ring(data_size)
            nrg = np.random.choice(self.nrgs, size=data_size)

            features = np.concatenate([
                center[:, 0].reshape(-1, 1),
                center[:, 1].reshape(-1, 1),
                mean.reshape(-1, 1),
                sig.reshape(-1, 1),
                nrg.reshape(-1, 1)
            ], axis=1)
            
            np.save(features_path, features)


            prob_distr_scaled = prob_distr / prob_distr.sum(axis=1)[:, np.newaxis]
            pmt_hits = [np.random.choice(self.N2, size=nrg[i], p=prob_distr_scaled[i]) for i in range(data_size)]

            imgs = np.zeros((data_size, self.N2), dtype=np.bool)
            for i in range(data_size):
                imgs[i, pmt_hits[i]] = 1

            self.imgs = imgs.reshape((data_size, self.N, self.N))
            np.save(imgs_path, self.imgs)

        self.scaler = mmScaler()
        self.features = self.scaler.fit_transform(features)

        train_dataset = TensorDataset(
            torch.tensor(self.features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            torch.tensor(self.imgs[:int(self.train_fraction * data_size)], dtype=torch.float, device=device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            torch.tensor(self.imgs[int(self.train_fraction * data_size):], dtype=torch.float, device=device)
        )
        
        if test_return:
            return self.features, self.imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def gaussian_ring(self, data_size):
        center = self.centers[np.random.choice(self.N2, size=data_size)]
        mean = np.random.choice(self.means, size=data_size)
        sig = np.random.choice(self.sigs, size=data_size)

        radius = np.linalg.norm(self.img_coor[np.newaxis, :, :] - center[:, np.newaxis, :], axis=2)
        
        return (
            np.exp(-(radius - mean[:, np.newaxis])**2 / sig[:, np.newaxis]**2),
            center,
            mean,
            sig
        )

    def prob_distr(self, pmts=None, centers=None, means=None, sigs=None, nrgs=None):
        variables = ()
        parameters = ()
        if pmts is None:
            pmts = np.array([self.centers[np.random.choice(self.N2)]])
            parameters += (pmts[0],)
        elif pmts.shape[0] > 1:
            variables += (pmts,)
        else:
            parameters += (pmts[0],)

        if centers is None:
            centers = np.array([self.centers[np.random.choice(self.N2)]])
            parameters += (centers[0],)
        elif centers.shape[0] > 1:
            variables += (centers,)
        else:
            parameters += (centers[0],)

        if means is None:
            means = np.array([np.random.choice(self.means)])
            parameters += (means[0],)
        elif len(means) > 1:
            variables += (means,)
        else:
            parameters += (means[0],)
            
        if sigs is None:
            sigs = np.array([np.random.choice(self.sigs)])
            parameters += (sigs[0],)
        elif len(sigs) > 1:
            variables += (sigs,)
        else:
            parameters += (sigs[0],)
            
        if nrgs is None:
            nrgs = np.array([np.random.choice(self.nrgs)])
            parameters += (nrgs[0],)
        elif len(nrgs) > 1:
            variables += (nrgs,)
        else:
            parameters += (nrgs[0],)

        features = np.zeros((centers.shape[0] * means.size * sigs.size * nrgs.size, self.n_features))

        distr = np.zeros((pmts.shape[0], centers.shape[0], means.size, sigs.size, nrgs.size))
        n = 0
        for i, pmt in enumerate(pmts):
            for j, center in enumerate(centers):
                for k, mean in enumerate(means):
                    for l, sig in enumerate(sigs):
                        for m, nrg in enumerate(nrgs):
                            distr[i, j, k, l, m] = np.exp(-(np.linalg.norm(pmt - center) - mean)**2 / sig**2) * nrg / self.N2
                            
                            features[n] = [center[0], center[1], mean, sig, nrg]
                            n += 1

        distr = distr.squeeze()
        features = self.scaler.transform(features)

        return distr, variables, parameters, features

    def gaussian_from_features(self, center, mean, sig):

        radius = np.linalg.norm(self.img_coor - center, axis=1)
        
        return np.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N))



class energy_randomized_ring_dataset():
    def __init__(self, N=32, train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.train_fraction = train_fraction

        self.centers = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)
        self.means = np.arange(N * 0.2, N * 0.4, N / 32)
        self.sigs = np.arange(N * 0.05, N * 0.2, N / 32)

        self.img_coor = np.array([(i, j) for i in range(self.N) for j in range(self.N)], dtype=np.int32)
        self.n_features = 4

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu', scale_img=True, test_return=False):
        features_path = f'../save_dataset/energy_features_N-{self.N}_data_size-{data_size}_seed-{seed}.npy'
        imgs_path = f'../save_dataset/energy_images_N-{self.N}_data_size-{data_size}_seed-{seed}.npy'
        if os.path.exists(features_path):
            features = np.load(features_path)
            self.imgs = np.load(imgs_path)

        else:
            np.random.seed(seed)

            gaus_ring, center, mean, sig = self.gaussian_ring(data_size)

            features = np.concatenate([
                center[:, 0].reshape(-1, 1),
                center[:, 1].reshape(-1, 1),
                mean.reshape(-1, 1),
                sig.reshape(-1, 1)
            ], axis=1)
            
            np.save(features_path, features)

            val = np.arange(0.7, 1.3, 0.01)
            distr = np.exp(-(val - 1)**2 / 0.2**2)

            kernel = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)])

            imgs = gaus_ring * kernel # adds gaussian noise

            self.img_scaler = mmScaler()
            imgs = self.img_scaler.fit_transform(imgs)

            self.imgs = imgs.reshape((data_size, self.N, self.N))
            np.save(imgs_path, self.imgs)

        self.scaler = mmScaler()
        self.features = self.scaler.fit_transform(features)

        train_dataset = TensorDataset(
            torch.tensor(self.features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            torch.tensor(self.imgs[:int(self.train_fraction * data_size)], dtype=torch.float, device=device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            torch.tensor(self.imgs[int(self.train_fraction * data_size):], dtype=torch.float, device=device)
        )
        
        if test_return:
            return self.features, self.imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def gaussian_ring(self, data_size):
        center = self.centers[np.random.choice(self.N2, size=data_size)]
        mean = np.random.choice(self.means, size=data_size)
        sig = np.random.choice(self.sigs, size=data_size)

        radius = np.linalg.norm(self.img_coor[np.newaxis, :, :] - center[:, np.newaxis, :], axis=2)
        
        return (
            np.exp(-(radius - mean[:, np.newaxis])**2 / sig[:, np.newaxis]**2),
            center,
            mean,
            sig
        )

    def gaussian_from_features(self, center_x, center_y, mean, sig):

        radius = np.linalg.norm(self.img_coor - (center_x, center_y), axis=1)
        
        return np.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N))


