import os

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler as mmScaler
from sklearn.preprocessing import PolynomialFeatures



class Ring():
    def __init__(self, N, save_path, inner_scale=0.1, outer_scale=0.45, ring_resolution=1, train_fraction=0.8):
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



class RandomRing():
    def __init__(self, N, save_path, train_fraction=0.8):
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



class PatternRandomRing():
    def __init__(self, N, save_path='../save_data', train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.save_path = save_path
        self.train_fraction = train_fraction

        if not os.path.exists(f'{self.save_path}/{self.__class__.__name__}/'):
            os.mkdir(f'{self.save_path}/{self.__class__.__name__}/')

        self.centers = np.array([(i, j) for i in range(8, N-8) for j in range(8, N-8)], dtype=np.int32)
        self.means = np.arange(6, 9, 0.03)
        self.sigs = np.arange(0.8, 1.6, 0.01)
        self.thetas = np.arange(0, 2 * np.pi, np.pi / 80)
        self.phis = np.arange(0, 0.5, 0.005)

        self.img_coor = np.array([(i, j) for i in range(self.N) for j in range(self.N)], dtype=np.int32)
        self.n_features = 6

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu', scale_img=True, test_return=False):
        features_path = f'{self.save_path}/{self.__class__.__name__}/features_{self.N}_{data_size}_{seed}.npy'
        imgs_path = f'{self.save_path}/{self.__class__.__name__}/images_{self.N}_{data_size}_{seed}.npy'
        if os.path.exists(features_path):
            features = np.load(features_path)
            self.imgs = np.load(imgs_path)

        else:
            np.random.seed(seed)

            # first particle
            center_1 = self.centers[np.random.choice(16**2, size=data_size)]
            center_x_1 = center_1[:, 0]
            center_y_1 = center_1[:, 1]
            mean_1 = np.random.choice(self.means, size=data_size)
            sig_1 = np.random.choice(self.sigs, size=data_size)
            theta_1 = np.random.choice(self.thetas, size=data_size)
            phi_1 = np.random.choice(self.phis, size=data_size)

            # second particle            
            center_2 = self.centers[np.random.choice(16**2, size=data_size)]
            center_x_2 = center_2[:, 0]
            center_y_2 = center_2[:, 1]
            mean_2 = np.random.choice(self.means, size=data_size)
            sig_2 = np.random.choice(self.sigs, size=data_size)
            theta_2 = np.random.choice(self.thetas, size=data_size)
            phi_2 = np.random.choice(self.phis, size=data_size)
            
            features = np.concatenate([
                center_x_1.reshape(-1, 1),
                center_y_1.reshape(-1, 1),
                mean_1.reshape(-1, 1),
                sig_1.reshape(-1, 1),
                theta_1.reshape(-1, 1),
                phi_1.reshape(-1, 1),
                center_x_2.reshape(-1, 1),
                center_y_2.reshape(-1, 1),
                mean_2.reshape(-1, 1),
                sig_2.reshape(-1, 1),
                theta_2.reshape(-1, 1),
                phi_2.reshape(-1, 1)
            ], axis=1)
            # np.save(features_path, features)

            imgs_1 = self.gaussian_rings(data_size, center_1, mean_1, sig_1, theta_1, phi_1)
            imgs_2 = self.gaussian_rings(data_size, center_2, mean_2, sig_2, theta_2, phi_2)

            val = np.arange(0, 2, 0.01)
            distr = np.exp(-(val - 1)**2 / 0.3**2)
            kernel = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)]) # adds gaussian noise

            particle_2 = np.array([np.random.choice([0, 1]) for _ in range(data_size)]) # chooses if there are 2 particles

            self.imgs = ((imgs_1 + imgs_2 * particle_2[:, np.newaxis]) * kernel).reshape((data_size, self.N, self.N))
            # np.save(imgs_path, self.imgs)

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

    def gaussian_rings(self, data_size, center, mean, sig, theta, phi):
        centered = self.img_coor[np.newaxis, :, :] - center[:, np.newaxis, :]
        radius = np.linalg.norm(centered, axis=2)

        angle = np.arccos(centered[:, :, 0] / (np.sqrt(centered[:, :, 1]**2 + centered[:, :, 0]**2) + 1e-10)) * np.sign(centered[:, :, 1] + 1e-10)
        radius += np.cos(angle - theta[:, np.newaxis]) * radius * phi[:, np.newaxis]

        return np.exp(-(radius - mean[:, np.newaxis])**2 / sig[:, np.newaxis]**2)

    def gaussian_from_features(self, center_x, center_y, mean, sig, theta, phi):
        centered = self.img_coor - (center_x, center_y)
        radius = np.linalg.norm(centered, axis=1)

        angle = np.arccos(centered[:, 0] / (np.sqrt(centered[:, 1]**2 + centered[:, 0]**2) + 1e-10)) * np.sign(centered[:, 1] + 1e-10)
        radius += np.cos(angle - theta) * radius * phi
        
        return np.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N))



class MultiRandomRing():
    def __init__(self, N, save_path='../save_data', train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.save_path = save_path
        self.train_fraction = train_fraction

        if not os.path.exists(f'{self.save_path}/{self.__class__.__name__}/'):
            os.mkdir(f'{self.save_path}/{self.__class__.__name__}/')

        self.centers = np.array([(i, j) for i in range(8, N-8) for j in range(8, N-8)])
        self.means = np.arange(6, 9, 0.03)
        self.sigs = np.arange(0.8, 1.6, 0.01)
        self.thetas = np.arange(0, 2 * np.pi, np.pi / 80)
        self.phis = np.arange(0, 0.5, 0.005)

        self.img_coor = np.array([(i, j) for i in range(self.N) for j in range(self.N)])
        self.n_features = 12

    def generate_dataset(self, data_size=10_000, batch_size=64, seed=42, device='cpu', test_return=False):
        features_path = f'{self.save_path}/{self.__class__.__name__}/features_{self.N}_{data_size}_{seed}.npy'
        imgs_path = f'{self.save_path}/{self.__class__.__name__}/images_{self.N}_{data_size}_{seed}.npy'
        if os.path.exists(features_path):
            features = np.load(features_path)
            self.imgs = np.load(imgs_path)

        else:
            np.random.seed(seed)

            # first particle
            center_1 = self.centers[np.random.choice(16**2, size=data_size)]
            center_x_1 = center_1[:, 0]
            center_y_1 = center_1[:, 1]
            mean_1 = np.random.choice(self.means, size=data_size)
            sig_1 = np.random.choice(self.sigs, size=data_size)
            theta_1 = np.random.choice(self.thetas, size=data_size)
            phi_1 = np.random.choice(self.phis, size=data_size)

            # second particle            
            center_2 = self.centers[np.random.choice(16**2, size=data_size)]
            center_x_2 = center_2[:, 0]
            center_y_2 = center_2[:, 1]
            mean_2 = np.random.choice(self.means, size=data_size)
            sig_2 = np.random.choice(self.sigs, size=data_size)
            theta_2 = np.random.choice(self.thetas, size=data_size)
            phi_2 = np.random.choice(self.phis, size=data_size)
            
            imgs_1 = self.gaussian_rings(data_size, center_1, mean_1, sig_1, theta_1, phi_1)
            imgs_2 = imgs_1.copy()
            np.random.shuffle(imgs_2)

            val = np.arange(0, 2, 0.01)
            distr = np.exp(-(val - 1)**2 / 0.3**2)
            kernel = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)]) # adds gaussian noise

            particle_choice = {0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]}
            particles = np.array([particle_choice[np.random.choice(4, p=[1/12, 1/3, 1/3, 1/4])] for _ in range(data_size)])

            features = np.concatenate([
                center_x_1.reshape(-1, 1),
                center_y_1.reshape(-1, 1),
                mean_1.reshape(-1, 1),
                sig_1.reshape(-1, 1),
                theta_1.reshape(-1, 1),
                phi_1.reshape(-1, 1),
                center_x_2.reshape(-1, 1),
                center_y_2.reshape(-1, 1),
                mean_2.reshape(-1, 1),
                sig_2.reshape(-1, 1),
                theta_2.reshape(-1, 1),
                phi_2.reshape(-1, 1)
            ], axis=1)
            features[:, :6] *= particles[:, np.newaxis, 0]
            features[:, 6:] *= particles[:, np.newaxis, 1]
            if not test_return:
                np.save(features_path, features)

            self.imgs = ((imgs_1 * particles[:, np.newaxis, 0] + imgs_2 * particles[:, np.newaxis, 1]) * kernel).reshape((data_size, self.N, self.N)) / 4
            if not test_return:
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

    def gaussian_rings(self, data_size, center, mean, sig, theta, phi):
        centered = self.img_coor[np.newaxis, :, :] - center[:, np.newaxis, :]
        radius = np.linalg.norm(centered, axis=2)

        angle = np.arccos(centered[:, :, 0] / (np.sqrt(centered[:, :, 1]**2 + centered[:, :, 0]**2) + 1e-6)) * np.sign(centered[:, :, 1] + 1e-6)
        radius += np.cos(angle - theta[:, np.newaxis]) * radius * phi[:, np.newaxis]

        return np.exp(-(radius - mean[:, np.newaxis])**2 / sig[:, np.newaxis]**2)

    def gaussian_from_features(self,
        center_x_1,
        center_y_1,
        mean_1,
        sig_1,
        theta_1,
        phi_1,
        center_x_2,
        center_y_2,
        mean_2,
        sig_2,
        theta_2,
        phi_2
    ):
        img_1 = self._gaussian_from_features(center_x_1, center_y_1, mean_1, sig_1, theta_1, phi_1)
        img_2 = self._gaussian_from_features(center_x_2, center_y_2, mean_2, sig_2, theta_2, phi_2)

        return (img_1 + img_2) / 4
    
    def _gaussian_from_features(self, center_x, center_y, mean, sig, theta, phi):
        centered = self.img_coor - (center_x, center_y)
        radius = np.linalg.norm(centered, axis=1)

        angle = np.arccos(centered[:, 0] / (np.sqrt(centered[:, 1]**2 + centered[:, 0]**2) + 1e-6)) * np.sign(centered[:, 1] + 1e-6)
        radius += np.cos(angle - theta) * radius * phi
        
        return np.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N))



class SingleRandomRing():
    def __init__(self, N, save_path='../save_data', train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.save_path = save_path
        self.train_fraction = train_fraction

        if not os.path.exists(f'{self.save_path}/{self.__class__.__name__}/'):
            os.mkdir(f'{self.save_path}/{self.__class__.__name__}/')

        self.centers = torch.Tensor([(i, j) for i in range(8, N-8) for j in range(8, N-8)])

        self.img_coor = torch.Tensor([(i, j) for i in range(self.N) for j in range(self.N)])
        self.n_features = 6

    def generate_dataset(self, data_size=10_000, batch_size=64, noise=True, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        if noise:
            noise_tag = f'noise_{sigma}_'
            # val = np.arange(0, 2, 0.01)
            # distr = np.exp(-(val - 1)**2 / sigma**2)
            # self.kernel = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)]) # adds gaussian noise
            # self.kernel_bis = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)])

            torch.manual_seed(seed)
            mean = torch.full((data_size, self.N2), 1.)
            std = torch.full((data_size, self.N2), sigma)
            self.kernel = torch.normal(mean, std)
            self.kernel_bis = torch.normal(mean, std)

        else:
            noise_tag = ''

        features_path = f'{self.save_path}/{self.__class__.__name__}/features_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        true_imgs_path = f'{self.save_path}/{self.__class__.__name__}/true_images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        imgs_path = f'{self.save_path}/{self.__class__.__name__}/images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        noise_delta_path = f'{self.save_path}/{self.__class__.__name__}/noise_delta_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        
        if os.path.exists(features_path):
            self.features = torch.load(features_path)
            # self.imgs_1 = torch.load(true_imgs_path)
            self.imgs = torch.load(imgs_path)
            self.noise_delta = torch.load(noise_delta_path)

        else:
            # first particle
            center_x_1 = torch.randint(8, 16, (data_size,))
            center_y_1 = torch.randint(8, 16, (data_size,))
            mean_1 = torch.rand((data_size,)) * 3 + 6
            sig_1 = torch.rand((data_size,)) * 0.8 + 0.8
            theta_1 = torch.zeros((data_size,)) * 2 * np.pi
            phi_1 = torch.zeros((data_size,)) * 0.5
            
            self.features = torch.concat([
                center_x_1.reshape(-1, 1),
                center_y_1.reshape(-1, 1),
                mean_1.reshape(-1, 1),
                sig_1.reshape(-1, 1),
                theta_1.reshape(-1, 1),
                phi_1.reshape(-1, 1)
            ], axis=1)

            center_1 = torch.concat([center_x_1.reshape(-1, 1), center_y_1.reshape(-1, 1)], dim=1)
            self.imgs_1 = self.gaussian_rings(data_size, center_1, mean_1, sig_1, theta_1, phi_1)

            self.noise_delta = {'mse_loss':{'mean':0., 'std':0.}, 'l1_loss':{'mean':0., 'std':0.}}
            if noise:
                self.imgs = (self.imgs_1 * self.kernel)
                self.imgs_bis = (self.imgs_1 * self.kernel_bis)

                self.noise_delta['l1_loss']['mean'] = torch.abs(self.imgs - self.imgs_bis).mean(dim=1)
                self.noise_delta['l1_loss']['std'] = self.noise_delta['l1_loss']['mean'].std().item()
                self.noise_delta['l1_loss']['mean'] = self.noise_delta['l1_loss']['mean'].mean().item()

                self.noise_delta['mse_loss']['mean'] = ((self.imgs - self.imgs_bis)**2).mean(dim=1)
                self.noise_delta['mse_loss']['std'] = self.noise_delta['mse_loss']['mean'].std().item()
                self.noise_delta['mse_loss']['mean'] = self.noise_delta['mse_loss']['mean'].mean().item()

                self.imgs = self.imgs.reshape((data_size, self.N, self.N))

            else:
                self.imgs = self.imgs_1.reshape((data_size, self.N, self.N))
                
            if not test_return:
                torch.save(self.features, features_path)
                # torch.save(self.imgs_1, true_imgs_path)
                torch.save(self.imgs, imgs_path)
                torch.save(self.noise_delta, noise_delta_path)

        self.poly = PolynomialFeatures(degree=features_degree)
        transformed_features = self.poly.fit_transform(self.features)

        self.scaler = mmScaler()
        self.transformed_features = self.scaler.fit_transform(transformed_features)

        train_dataset = TensorDataset(
            torch.tensor(self.transformed_features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            self.imgs[:int(self.train_fraction * data_size)].to(device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.transformed_features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            self.imgs[int(self.train_fraction * data_size):].to(device)
        )
        
        if test_return:
            return self.features, self.imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def generate_pixel_dataset(self, data_size=10_000, batch_size=64, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        self.generate_dataset(data_size=data_size, noise=True, sigma=sigma, features_degree=features_degree, seed=seed, device=device, test_return=False)

        self.features

    def gaussian_rings(self, data_size, center, mean, sig, theta, phi):
        centered = self.img_coor[None, :, :] - center[:, None, :]
        radius = torch.linalg.norm(centered, dim=2)

        angle = torch.acos(centered[:, :, 0] / (torch.sqrt(centered[:, :, 1]**2 + centered[:, :, 0]**2) + 1e-6)) * torch.sign(centered[:, :, 1] + 1e-6)
        radius += torch.cos(angle - theta[:, None]) * radius * phi[:, None]

        return torch.exp(-(radius - mean[:, None])**2 / sig[:, None]**2) / 2

    def random_sample(self, n_samples: int):
        data_size = len(self.features)
        idx = torch.randint(0, data_size, (n_samples,))

        features = self.features[idx]
        transformed_features = self.scaler.transform(self.poly.transform(features))
        imgs = self.imgs[idx]

        return features.numpy(), transformed_features, imgs.numpy()

    def ring_from_features(self, center_x, center_y, mean, sig, theta, phi):
        centered = self.img_coor - torch.tensor([center_x, center_y])
        radius = torch.linalg.norm(centered, dim=1)

        angle = torch.acos(centered[:, 0] / (torch.sqrt(centered[:, 1]**2 + centered[:, 0]**2) + 1e-6)) * torch.sign(centered[:, 1] + 1e-6)
        radius += torch.cos(angle - theta) * radius * phi
        
        return torch.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N)) / 2



class NoiseRing():
    def __init__(self, N, save_path='../save_data', train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.save_path = save_path
        self.train_fraction = train_fraction

        if not os.path.exists(f'{self.save_path}/{self.__class__.__name__}/'):
            os.mkdir(f'{self.save_path}/{self.__class__.__name__}/')

        self.centers = torch.Tensor([(i, j) for i in range(8, N-8) for j in range(8, N-8)])

        self.img_coor = torch.Tensor([(i, j) for i in range(self.N) for j in range(self.N)])
        self.n_features = 6

    def generate_dataset(self, data_size=10_000, batch_size=64, noise=True, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        if noise:
            noise_tag = f'noise_{sigma}_'
            # val = np.arange(0, 2, 0.01)
            # distr = np.exp(-(val - 1)**2 / sigma**2)
            # self.kernel = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)]) # adds gaussian noise
            # self.kernel_bis = np.array([np.random.choice(val, size=self.N2, p=distr / distr.sum()) for _ in range(data_size)])

            torch.manual_seed(seed)
            mean = torch.full((data_size, self.N2), 1.)
            std = torch.full((data_size, self.N2), sigma)
            self.kernel = torch.normal(mean, std)
            self.kernel_bis = torch.normal(mean, std)

        else:
            noise_tag = ''

        features_path = f'{self.save_path}/{self.__class__.__name__}/features_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        true_imgs_path = f'{self.save_path}/{self.__class__.__name__}/true_images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        imgs_path = f'{self.save_path}/{self.__class__.__name__}/images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        noise_delta_path = f'{self.save_path}/{self.__class__.__name__}/noise_delta_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        
        if os.path.exists(features_path):
            self.features = torch.load(features_path)
            self.imgs_1 = torch.load(true_imgs_path)
            self.imgs = torch.load(imgs_path)
            self.noise_delta = torch.load(noise_delta_path)

        else:
            # first particle
            center_x_1 = torch.randint(8, 16, (data_size,))
            center_y_1 = torch.randint(8, 16, (data_size,))
            mean_1 = torch.rand((data_size,)) * 3 + 6
            sig_1 = torch.rand((data_size,)) * 0.8 + 0.8
            theta_1 = torch.rand((data_size,)) * 2 * np.pi
            phi_1 = torch.rand((data_size,)) * 0.5
            
            self.features = torch.concat([
                center_x_1.reshape(-1, 1),
                center_y_1.reshape(-1, 1),
                mean_1.reshape(-1, 1),
                sig_1.reshape(-1, 1),
                theta_1.reshape(-1, 1),
                phi_1.reshape(-1, 1)
            ], axis=1)

            center_1 = torch.concat([center_x_1.reshape(-1, 1), center_y_1.reshape(-1, 1)], dim=1)
            self.imgs_1 = self.gaussian_rings(data_size, center_1, mean_1, sig_1, theta_1, phi_1)

            self.noise_delta = {'mse_loss':{'mean':0., 'std':0.}, 'l1_loss':{'mean':0., 'std':0.}}
            if noise:
                self.imgs = (self.imgs_1 * self.kernel) - self.imgs_1
                self.imgs_bis = (self.imgs_1 * self.kernel_bis) - self.imgs_1

                self.noise_delta['l1_loss']['mean'] = torch.abs(self.imgs - self.imgs_bis).mean(dim=1)
                self.noise_delta['l1_loss']['std'] = self.noise_delta['l1_loss']['mean'].std().item()
                self.noise_delta['l1_loss']['mean'] = self.noise_delta['l1_loss']['mean'].mean().item()

                self.noise_delta['mse_loss']['mean'] = ((self.imgs - self.imgs_bis)**2).mean(dim=1)
                self.noise_delta['mse_loss']['std'] = self.noise_delta['mse_loss']['mean'].std().item()
                self.noise_delta['mse_loss']['mean'] = self.noise_delta['mse_loss']['mean'].mean().item()

                self.imgs = self.imgs.reshape((data_size, self.N, self.N))

            else:
                self.imgs = self.imgs_1.reshape((data_size, self.N, self.N))
                
            if not test_return:
                torch.save(self.features, features_path)
                torch.save(self.imgs_1, true_imgs_path)
                torch.save(self.imgs, imgs_path)
                torch.save(self.noise_delta, noise_delta_path)

        self.poly = PolynomialFeatures(degree=features_degree)
        transformed_features = self.poly.fit_transform(self.features)

        self.scaler = mmScaler()
        self.transformed_features = self.scaler.fit_transform(transformed_features)

        train_dataset = TensorDataset(
            torch.tensor(self.transformed_features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            self.imgs[:int(self.train_fraction * data_size)].to(device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.transformed_features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            self.imgs[int(self.train_fraction * data_size):].to(device)
        )
        
        if test_return:
            return self.features, self.imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def generate_pixel_dataset(self, data_size=10_000, batch_size=64, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        self.generate_dataset(data_size=data_size, noise=True, sigma=sigma, features_degree=features_degree, seed=seed, device=device, test_return=False)

        self.features

    def gaussian_rings(self, data_size, center, mean, sig, theta, phi):
        centered = self.img_coor[None, :, :] - center[:, None, :]
        radius = torch.linalg.norm(centered, dim=2)

        angle = torch.acos(centered[:, :, 0] / (torch.sqrt(centered[:, :, 1]**2 + centered[:, :, 0]**2) + 1e-6)) * torch.sign(centered[:, :, 1] + 1e-6)
        radius += torch.cos(angle - theta[:, None]) * radius * phi[:, None]

        return torch.exp(-(radius - mean[:, None])**2 / sig[:, None]**2) / 2

    def random_sample(self, n_samples: int):
        data_size = len(self.features)
        idx = torch.randint(0, data_size, (n_samples,))

        features = self.features[idx]
        transformed_features = self.scaler.transform(self.poly.transform(features))
        imgs = self.imgs[idx]

        return features.numpy(), transformed_features, imgs.numpy()

    def ring_from_features(self, center_x, center_y, mean, sig, theta, phi):
        centered = self.img_coor - torch.tensor([center_x, center_y])
        radius = torch.linalg.norm(centered, dim=1)

        angle = torch.acos(centered[:, 0] / (torch.sqrt(centered[:, 1]**2 + centered[:, 0]**2) + 1e-6)) * torch.sign(centered[:, 1] + 1e-6)
        radius += torch.cos(angle - theta) * radius * phi
        
        return torch.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N)) / 2



class PixelRing():
    def __init__(self, N, save_path='../save_data', train_fraction=0.8):
        self.N = N
        self.N2 = N**2
        self.save_path = save_path
        self.train_fraction = train_fraction

        if not os.path.exists(f'{self.save_path}/{self.__class__.__name__}/'):
            os.mkdir(f'{self.save_path}/{self.__class__.__name__}/')

        self.centers = torch.Tensor([(i, j) for i in range(8, N-8) for j in range(8, N-8)])

        self.img_coor = torch.Tensor([(i, j) for i in range(self.N) for j in range(self.N)])
        self.n_features = 6

    def generate_dataset(self, data_size=10_000, batch_size=64, noise=True, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        if noise:
            noise_tag = f'noise_{sigma}_'

            torch.manual_seed(seed)
            mean = torch.full((data_size, self.N2), 1.)
            std = torch.full((data_size, self.N2), sigma)
            self.kernel = torch.normal(mean, std)
            self.kernel_bis = torch.normal(mean, std)

        else:
            noise_tag = ''

        features_path = f'{self.save_path}/{self.__class__.__name__}/features_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        true_imgs_path = f'{self.save_path}/{self.__class__.__name__}/true_images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        imgs_path = f'{self.save_path}/{self.__class__.__name__}/images_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        noise_delta_path = f'{self.save_path}/{self.__class__.__name__}/noise_delta_{self.N}_{data_size}_{noise_tag}{features_degree}_{seed}.npy'
        
        if os.path.exists(features_path):
            self.features = torch.load(features_path)
            self.imgs_1 = torch.load(true_imgs_path)
            self.imgs = torch.load(imgs_path)
            self.noise_delta = torch.load(noise_delta_path)

        else:
            # first particle
            center_x_1 = torch.randint(8, 16, (data_size,))
            center_y_1 = torch.randint(8, 16, (data_size,))
            mean_1 = torch.rand((data_size,)) * 3 + 6
            sig_1 = torch.rand((data_size,)) * 0.8 + 0.8
            theta_1 = torch.zeros((data_size,)) * 2 * np.pi
            phi_1 = torch.zeros((data_size,)) * 0.5
            
            self.features = torch.concat([
                center_x_1.reshape(-1, 1),
                center_y_1.reshape(-1, 1),
                mean_1.reshape(-1, 1),
                sig_1.reshape(-1, 1),
                theta_1.reshape(-1, 1),
                phi_1.reshape(-1, 1)
            ], axis=1)

            center_1 = torch.concat([center_x_1.reshape(-1, 1), center_y_1.reshape(-1, 1)], dim=1)
            self.imgs_1 = self.gaussian_rings(data_size, center_1, mean_1, sig_1, theta_1, phi_1)

            self.noise_delta = {'mse_loss':{'mean':0., 'std':0.}, 'l1_loss':{'mean':0., 'std':0.}}
            if noise:
                self.imgs = (self.imgs_1 * self.kernel)
                self.imgs_bis = (self.imgs_1 * self.kernel_bis)

                self.noise_delta['l1_loss']['mean'] = torch.abs(self.imgs - self.imgs_bis).mean(dim=1)
                self.noise_delta['l1_loss']['std'] = self.noise_delta['l1_loss']['mean'].std().item()
                self.noise_delta['l1_loss']['mean'] = self.noise_delta['l1_loss']['mean'].mean().item()

                self.noise_delta['mse_loss']['mean'] = ((self.imgs - self.imgs_bis)**2).mean(dim=1)
                self.noise_delta['mse_loss']['std'] = self.noise_delta['mse_loss']['mean'].std().item()
                self.noise_delta['mse_loss']['mean'] = self.noise_delta['mse_loss']['mean'].mean().item()

            else:
                self.imgs = self.imgs_1.clone()
                
            if not test_return:
                torch.save(self.features, features_path)
                torch.save(self.imgs_1, true_imgs_path)
                torch.save(self.imgs, imgs_path)
                torch.save(self.noise_delta, noise_delta_path)

        self.poly = PolynomialFeatures(degree=features_degree)
        transformed_features = self.poly.fit_transform(self.features)

        self.scaler = mmScaler()
        self.transformed_features = self.scaler.fit_transform(transformed_features)

        train_dataset = TensorDataset(
            torch.tensor(self.transformed_features[:int(self.train_fraction * data_size)], dtype=torch.float, device=device),
            self.imgs[:int(self.train_fraction * data_size)].to(device),
            self.imgs_1[:int(self.train_fraction * data_size)].to(device)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.transformed_features[int(self.train_fraction * data_size):], dtype=torch.float, device=device),
            self.imgs[int(self.train_fraction * data_size):].to(device),
            self.imgs_1[int(self.train_fraction * data_size):].to(device)
        )
        
        if test_return:
            return self.features, self.imgs
        else:
            return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    def generate_pixel_dataset(self, data_size=10_000, batch_size=64, sigma=0.3, features_degree=1, seed=42, device='cpu', test_return=False):
        self.generate_dataset(data_size=data_size, noise=True, sigma=sigma, features_degree=features_degree, seed=seed, device=device, test_return=False)

        self.features

    def gaussian_rings(self, data_size, center, mean, sig, theta, phi):
        centered = self.img_coor[None, :, :] - center[:, None, :]
        radius = torch.linalg.norm(centered, dim=2)

        angle = torch.acos(centered[:, :, 0] / (torch.sqrt(centered[:, :, 1]**2 + centered[:, :, 0]**2) + 1e-6)) * torch.sign(centered[:, :, 1] + 1e-6)
        radius += torch.cos(angle - theta[:, None]) * radius * phi[:, None]

        return torch.exp(-(radius - mean[:, None])**2 / sig[:, None]**2) / 2

    def random_sample(self, n_samples: int):
        data_size = len(self.features)
        idx = torch.randint(0, data_size, (n_samples,))

        features = self.features[idx]
        transformed_features = self.scaler.transform(self.poly.transform(features))
        imgs = self.imgs[idx]

        return features.numpy(), transformed_features, imgs.numpy()

    def ring_from_features(self, center_x, center_y, mean, sig, theta, phi):
        centered = self.img_coor - torch.tensor([center_x, center_y])
        radius = torch.linalg.norm(centered, dim=1)

        angle = torch.acos(centered[:, 0] / (torch.sqrt(centered[:, 1]**2 + centered[:, 0]**2) + 1e-6)) * torch.sign(centered[:, 1] + 1e-6)
        radius += torch.cos(angle - theta) * radius * phi
        
        return torch.exp(-(radius - mean)**2 / sig**2).reshape((self.N, self.N)) / 2


