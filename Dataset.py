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
        try:
            source_shape_image = Image.open(image_path).convert('RGB')
        except:
            print(image_path, 'does not exist')
            return None, None
        if self.transform:
            source_shape_image = self.transform(source_shape_image)
        return source_shape_image, source_shape_image

def load_dataset(split_name: str):
    assert split_name in ['train', 'val', 'test']
    split_dir = os.path.join('Dataset Splits', split_name)
    file_paths = np.load(os.path.join(split_dir, split_name + '.npy'), allow_pickle=True)
    split_transform = get_split_transforms(split_name)
    split_dataset = ShapeNetMultiViewDataset(file_paths.tolist(), transform=split_transform)
    return split_dataset

def save_dataset(split_name: str, path_list: list):
    path_np = np.array(path_list, dtype=object)
    split_dir = os.path.join('Dataset Splits', split_name)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    np.save(os.path.join(split_dir, split_name + '.npy'), path_np)

def get_split_transforms(split_name: str):
    assert split_name in ['train', 'val', 'test']
    if split_name == 'train':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(augmentation)

    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return transform

def generate_datasets(dataset_path, portions=None, use_prev_indices=False):
    if use_prev_indices:
        datasets = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets[split_name] = load_dataset(split_name=split_name)
    else:
        if portions is None:
            portions = {'train': .5, 'val': .15, 'test': .1}
            # portions = {'train': .1, 'val': .03, 'test': .02}

        step_size_dict = {'train': max(1, int(1 / portions['train'])),
                    'val': max(1, int(1 / portions['val'])),
                    'test': max(1, int(1 / portions['test']))}
        dataset_split_file_paths = {'train': [], 'val': [], 'test': []}
        datasets = {'train': None, 'val': None, 'test': None}
        print('generating datasets...')
        data_categories_path_list = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if '.' not in x]
        data_models_dir_list = np.array([os.path.join(x, y) for x in data_categories_path_list for y in
                                      os.listdir(x) if '.' not in y], dtype=object)
        # data_models_path_list = [] #np.empty_like(data_models_dir_list, dtype=object)
        for ind, split_name in enumerate(['train', 'val', 'test']):
            step = step_size_dict[split_name]
            for i in range(data_models_dir_list.shape[0]):
                rotation_dir = os.path.join(str(data_models_dir_list[i]), 'models', '0')
                try:
                    all_image_names = sorted([name for name in os.listdir(rotation_dir) if name.endswith('.png')])
                    selected_files = all_image_names[ind::step]
                    for image_name in selected_files:
                        dataset_split_file_paths[split_name].append(os.path.join(rotation_dir, image_name))

                except:
                    print(rotation_dir, 'does not exist')
                    continue


        for split_name in ['train', 'val', 'test']:
            split_transform = get_split_transforms(split_name)
            datasets[split_name] = ShapeNetMultiViewDataset(dataset_split_file_paths[split_name], transform=split_transform)
            save_dataset(split_name, dataset_split_file_paths[split_name])
        print('train size: ', len(datasets['train']), 'val size: ', len(datasets['val']), 'test size: ', len(datasets['test']))
    return datasets