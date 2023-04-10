import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler as mmScaler



class simple_ring_dataset():
    def __init__(self, N=32, inner_scale=0.1, outer_scale=0.45, ring_resolution=1, train_fraction=0.8):
        self.N = N
        self.train_fraction = train_fraction
        self.centers = np.array([[(i, j) for i in range(N)] for j in range(N)], dtype=np.int32).reshape((N * N, 2))
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

    def random_ring(self):
        center = self.centers[np.random.choice(self.N * self.N)]
        radius = np.linalg.norm(self.img_coor - center, axis=2)

        width_idx = np.random.choice(len(self.inners))
        img = np.zeros_like(radius, dtype=np.bool)
        img[np.logical_and(radius >= self.inners[width_idx], radius < self.outers[width_idx])] = 1

        features = np.array([center[0], center[1], self.inners[width_idx], self.outers[width_idx]]).reshape(1, -1)
        features = torch.tensor(self.scaler.transform(features), dtype=torch.float)

        return features, img



class randomized_ring_dataset():
    def __init__(self, N=32, inner_scale=0.1, outer_scale=0.45, ring_resolution=1, train_fraction=0.8):
        self.N = N
        self.train_fraction = train_fraction
        self.centers = np.array([[(i, j) for i in range(N)] for j in range(N)], dtype=np.int32).reshape((N * N, 2))

        gaussians = []

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

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu'):
        np.random.seed(seed)

        centers_idx = np.random.choice(self.N * self.N, size=data_size)
        radius = np.linalg.norm(self.img_coor[np.newaxis, :, :, :] - self.centers[centers_idx, np.newaxis, np.newaxis, :], axis=3)

        mean = np.random.choice(np.arange(N * 0.2, N * 0.4), size=data_size)
        sig = np.random.choice(np.arange(N * 0.05, N * 0.3), size=data_size)

        ring_prob = gaus(radius, mean, sig)

        return ring_prob

def gaus(X, mean, sig):
    return np.exp(-(X - mean)**2 / sig**2)

N = 32
ring_prob = gaussian_distr_ring(32, 10000)
img_coor = np.array([[(i, j) for i in range(32)] for j in range(32)], dtype=np.int32)
activated_pmts = img_coor.reshape((N * N, 2))[np.random.choice(np.arange(N * N), 100, p=ring_prob.flatten() / ring_prob.sum())]
img = np.zeros_like(ring_prob, dtype=np.bool)
img[activated_pmts[:, 0], activated_pmts[:, 1]] = 1
Image.fromarray(img)







    def random_ring(self):
        center = self.centers[np.random.choice(self.N * self.N)]
        radius = np.linalg.norm(self.img_coor - center, axis=2)

        width_idx = np.random.choice(len(self.inners))
        img = np.zeros_like(radius, dtype=np.bool)
        img[np.logical_and(radius >= self.inners[width_idx], radius < self.outers[width_idx])] = 1

        features = np.array([center[0], center[1], self.inners[width_idx], self.outers[width_idx]]).reshape(1, -1)
        features = torch.tensor(self.scaler.transform(features), dtype=torch.float)

        return features, img
