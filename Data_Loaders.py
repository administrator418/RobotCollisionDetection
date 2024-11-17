import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        load_data = np.genfromtxt('saved/collect_data.csv', delimiter=',')
        X = load_data[:, :-1]
        y = load_data[:, -1]

        # balance the data
        mask = np.all(X[:, :5] == 150, axis=1)
        filtered_X = X[mask][:int(len(load_data) * 0.1)]
        filtered_y = y[mask][:int(len(load_data) * 0.1)]
        remaining_X = X[~mask]
        remaining_y = y[~mask]
        X_combined = np.vstack([remaining_X, filtered_X])
        y_combined = np.concatenate([remaining_y, filtered_y])

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_res, y_res = smote.fit_resample(X_combined, y_combined)

        balanced_data = np.column_stack((X_res, y_res))
        np.random.shuffle(balanced_data)

        # normalize data and save scaler for inference
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(balanced_data) # fits and transforms
        np.savetxt('saved/training_data.csv', self.data, delimiter=',', fmt='%.2f')
        pickle.dump(scaler, open("saved/scaler.pkl", "wb")) # save to normalize at inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()

        X = torch.tensor(self.data[idx][:-1], dtype=torch.float32)
        y = torch.tensor(self.data[idx][-1], dtype=torch.float32)

        return {'input': X, 'label': y}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()

        train_indices, test_indices = train_test_split(np.arange(len(self.nav_dataset)), test_size=0.2, random_state=42)

        train_subset = data.Subset(self.nav_dataset, train_indices)
        test_subset = data.Subset(self.nav_dataset, test_indices)

        self.train_loader = data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
