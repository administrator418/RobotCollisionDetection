import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle

from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        with FileLock('saved/collect_data.npy.lock'):
            load_data = np.load('saved/collect_data.npy')
        np.savetxt('saved/collect_data.csv', load_data, delimiter=',', fmt='%.2f')
        X = load_data[:, :-1]
        y = load_data[:, -1]

        # balance the data
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        balanced_data = np.column_stack((X_res, y_res))
        np.random.shuffle(balanced_data)

        # normalize data and save scaler for inference
        scaler = MinMaxScaler()
        pickle.dump(scaler, open("saved/scaler.pkl", "wb"))  # save to normalize at inference
        self.data = scaler.fit_transform(balanced_data) # fits and transforms
        np.savetxt('saved/training_data.csv', self.data, delimiter=',', fmt='%.2f')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()

        X = torch.tensor(self.data[idx][:-1], dtype=torch.float32)
        y = torch.tensor(self.data[idx][-1], dtype=torch.float32)

        return {'input': X, 'label': y}


class Data_Loaders():
    def __init__(self):
        self.nav_dataset = Nav_Dataset()

        train_indices, test_indices = train_test_split(np.arange(len(self.nav_dataset)), test_size=0.2, random_state=42)

        self.train_subset = data.Subset(self.nav_dataset, train_indices)
        self.test_subset = data.Subset(self.nav_dataset, test_indices)

    def get_train_data(self, batch_size):
        return data.DataLoader(self.train_subset, batch_size=batch_size, shuffle=True)

    def get_test_data(self, batch_size):
        return data.DataLoader(self.test_subset, batch_size=batch_size, shuffle=False)

def main():
    batch_size = 16
    data_loaders = Data_Loaders()
    
    for idx, sample in enumerate(data_loaders.get_train_data(batch_size)):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.get_test_data(batch_size)):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
