import torch
from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms.transforms as transforms
from torch.utils.data import DataLoader

class AddGaussianNoiseClipped:

    def __init__(self, mean = 0.0, std = 10.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img += self.std * torch.randn_like(img) + self.mean
        return torch.clamp(img, 0.0, 255.0)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SymbolDataset(Dataset):

    def __init__(self, data_folder, duplication_factor=1, eval=False) -> None:
        super().__init__()
        self.duplication_factor = duplication_factor

        abydos_folder = os.path.join(data_folder, "abydos")
        earth_folder = os.path.join(data_folder, "earth")

        self.data = []
        self.classifications = []

        for file in os.listdir(abydos_folder):
            read_img = cv2.imread(os.path.join(abydos_folder, file), cv2.IMREAD_GRAYSCALE)
            self.data.append(read_img)
            one_hot_encoding = torch.zeros(14, dtype=float)
            index = file.split('.')[0]
            one_hot_encoding[int(index)] = 1.0
            self.classifications.append(one_hot_encoding)

        for file in os.listdir(earth_folder):
            read_img = cv2.imread(os.path.join(earth_folder, file), cv2.IMREAD_GRAYSCALE)
            self.data.append(read_img)
            one_hot_encoding = torch.zeros(14, dtype=float)
            index = file.split('.')[0]
            one_hot_encoding[int(index)] = 1.0
            self.classifications.append(one_hot_encoding)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.RandomPerspective(distortion_scale=0.7, p=0.95),
            # AddGaussianNoiseClipped(std=0.1),
        ])

        self.eval_data = []
        self.eval = eval
        if self.eval:
            for _ in range(duplication_factor):
                for data in self.data:
                    self.eval_data.append(self.transform(data))

    def __len__(self):
        return len(self.data) * self.duplication_factor
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.eval:
            return self.eval_data[idx], self.classifications[idx % len(self.data)]

        idx = idx % len(self.data)
        data = self.data[idx]
        classification = self.classifications[idx]
        return self.transform(data), classification

# if __name__ == '__main__':
    # ds = SymbolDataset("data")        
    # first_elem = ds[1]
    # first_elem_array = first_elem[0].numpy()
    # first_elem_array = first_elem_array.reshape((64, 64))
    # cv2.imshow('hey', first_elem_array)
    # cv2.waitKey(0)
