import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler as mmScaler



class simple_ring_dataset():
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
        features = self.scaler.fit_transform(features)

        train_dataset = TensorDataset(
            torch.tensor(features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            torch.tensor(self.img[:int(self.train_fraction * data_size)], dtype=torch.float, device=device)
        )
        test_dataset = TensorDataset(
            torch.tensor(features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
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
    def __init__(self, N=32, ring_resolution=1, train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.train_fraction = train_fraction

        self.centers = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)
        self.means = np.arange(N * 0.2, N * 0.4, ring_resolution)
        self.sigs = np.arange(N * 0.05, N * 0.2, ring_resolution)
        self.nrgs = np.arange(0.05 * self.N2, 0.21 * self.N2, 0.02 * self.N2).astype(np.int32)

        self.img_coor = np.array([(i, j) for i in range(self.N) for j in range(self.N)], dtype=np.int32)
        self.n_features = 5

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu', test_return=False):
        np.random.seed(seed)

        prob_distr, center, mean, sig = self._gaussian_ring(data_size)
        nrg = np.random.choice(self.nrgs, size=data_size)

        prob_distr_scaled = prob_distr / prob_distr.sum(axis=1)[:, np.newaxis]
        pmt_hits = [np.random.choice(self.N2, size=nrg[i], p=prob_distr_scaled[i]) for i in range(data_size)]

        imgs = np.zeros((data_size, self.N2), dtype=np.bool)
        for i in range(data_size):
            imgs[i, pmt_hits[i]] = 1
        imgs = imgs.reshape((data_size, self.N, self.N))

        features = np.concatenate([
            center[:, 0].reshape(-1, 1),
            center[:, 1].reshape(-1, 1),
            mean.reshape(-1, 1),
            sig.reshape(-1, 1),
            nrg.reshape(-1, 1)
        ], axis=1)

        self.scaler = mmScaler()
        features = self.scaler.fit_transform(features)

        train_dataset = TensorDataset(
            torch.tensor(features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            torch.tensor(imgs[:int(self.train_fraction * data_size)], dtype=torch.float, device=device)
        )
        test_dataset = TensorDataset(
            torch.tensor(features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            torch.tensor(imgs[int(self.train_fraction * data_size):], dtype=torch.float, device=device)
        )
        
        if test_return:
            return imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def _gaussian_ring(self, data_size):
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

    def prob_distr(self, nb_samples=10000, centers=None, means=None, sigs=None, nrgs=None):
        returns = ()
        if centers is None:
            centers = [self.centers[np.random.choice(self.N2)]]
            returns += (centers[0],)

        if means is None:
            means = [np.random.choice(self.means)]
            returns += (means[0],)
            
        if sigs is None:
            sigs = [np.random.choice(self.sigs)]]
            returns += (sigs[0],)
            
        if nrgs is None:
            nrgs = [np.random.choice(self.nrgs)]]
            returns += (nrgs[0],)

        for center in centers:
            for mean in means:
                for sig in sigs:
                    for nrg in nrgs:



        return features, img
