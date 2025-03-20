from torch.utils import data
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

class ShapeNetMultiViewDataset(data.Dataset):
    def __init__(self, data_models_path_list, transform=None):
        self.data_models_path_list = data_models_path_list
        self.transform = transform

    def __len__(self):
        return len(self.data_models_path_list)

    def __getitem__(self, idx):
        image_path = self.data_models_path_list[idx]
        source_shape_image = Image.open(image_path).convert('RGB')
        target_shape_image = Image.open(image_path).convert('RGB')
        if self.transform:
            source_shape_image = self.transform(source_shape_image)
            target_shape_image = self.transform(target_shape_image)
        return source_shape_image, target_shape_image
    
def generate_datasets(dataset_path='./Sample Dataset', portions=None):
    if portions is None:
        portions = {'train': .75, 'val': .15, 'test': .1}
    data_categories_path_list = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if '.' not in x]
    data_models_dir_list = np.array([os.path.join(x, y) for x in data_categories_path_list for y in
                                  os.listdir(x) if '.' not in y], dtype=object)
    data_models_path_list = np.empty_like(data_models_dir_list, dtype=object)
    for i in range(data_models_dir_list.shape[0]):
        try:
            image_names = [name for name in os.listdir(os.path.join(str(data_models_dir_list[i]), 'models')) if name.endswith('.png')]
            for image_name in image_names:
                data_models_path_list[i] = os.path.join(str(data_models_dir_list[i]), 'models', image_name)
        except:
            print(os.path.join(str(data_models_dir_list[i]), 'models'), 'does not exist')
            continue
    train_size = int(len(data_models_path_list) * portions['train'])
    val_size = int(len(data_models_path_list) * portions['val'])
    test_size = int(len(data_models_path_list) * portions['test'])
    print('train size: ', train_size, 'val size: ', val_size, 'test size: ', test_size, '\n')

    permuted_indices = torch.randperm(len(data_models_path_list))
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:train_size + val_size]
    test_indices = permuted_indices[train_size + val_size:train_size + val_size + test_size]

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    train_trans = transforms.Compose(augmentation)

    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    train_dataset = ShapeNetMultiViewDataset(data_models_path_list[train_indices], transform=train_trans)
    val_dataset = ShapeNetMultiViewDataset(data_models_path_list[val_indices], transform=val_trans)
    test_dataset = ShapeNetMultiViewDataset(data_models_path_list[test_indices], transform=val_trans)

    return train_dataset, val_dataset, test_dataset