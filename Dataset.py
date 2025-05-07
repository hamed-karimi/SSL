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
            target_shape_image = Image.open(image_path).convert('RGB')
        except:
            print(image_path, 'does not exist')
            return None, None
        if self.transform:
            source_shape_image = self.transform(source_shape_image)
            target_shape_image = self.transform(target_shape_image)
        return source_shape_image, target_shape_image

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

def generate_datasets(dataset_path, portions=None, viewpoint_portions=None):
    if portions is None:
        portions = {'train': .5, 'val': .15, 'test': .1}
    if viewpoint_portions is None:
        viewpoint_portions = {'train': .1, 'val': .1, 'test': .1}

    step_size_dict = {'train': max(1, int(1 / viewpoint_portions['train'])),
                'val': max(1, int(1 / viewpoint_portions['val'])),
                'test': max(1, int(1 / viewpoint_portions['test']))}
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

            # remove this before commit
            # break

    for split_name in ['train', 'val', 'test']:
        split_transform = get_split_transforms(split_name)
        datasets[split_name] = ShapeNetMultiViewDataset(dataset_split_file_paths[split_name], transform=split_transform)
    print('train size: ', len(datasets['train']), 'val size: ', len(datasets['val']), 'test size: ', len(datasets['test']))
    return datasets