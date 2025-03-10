import torch.utils.data as Dataset
import torch
import os

class Dataset(Dataset):
    def __init__(self, data_path='./Sample Dataset'):
        self.data_path = data_path
        self.data_categories_path_list = os.listdir([os.path.join(self.data_path, x) for x in os.listdir(self.data_path)])
        self.data_models_path_list = os.listdir([os.path.join(self.data_path, x, y) for x in self.data_categories_path_list for y in os.listdir(x)])

    def __len__(self):
        return len(self.data_models_path_list)

    def __getitem__(self, idx):
        data_path = self.data_models_path_list[idx]
        data = torch.load(data_path)
        return data
    
            

