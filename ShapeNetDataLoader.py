import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# from Dataset import ShapeNetMultiViewDataset

def train_loader(train_dataset, parallel, batch_size, n_cpus) -> data.DataLoader:

    # [NO] do not use normalize here cause it's very hard to converge
    # [NO] do not use colorjitter cause it lead to performance drop in both train set and val set

    # [?] guassian blur will lead to a significantly drop in train loss while val loss remain the same

    if parallel == 1:
        n_gpus = torch.cuda.device_count()
        num_workers = n_cpus // n_gpus
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)         
    else:  
        train_sampler = None
        num_workers = n_cpus

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_dataloader

def val_loader(val_dataset, parallel, batch_size, n_cpus) -> data.DataLoader:

    if parallel == 1:
        n_gpus = torch.cuda.device_count()
        num_workers = n_cpus // n_gpus
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        val_sampler = None
        num_workers = n_cpus

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler)

    return val_dataloader
