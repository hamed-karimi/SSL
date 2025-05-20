import torch
# import torch.utils.data as data
import torchvision.transforms as transforms
# from Dataset import ShapeNetMultiViewDataset

def remove_none_indices(batch):
    # Filter out None values from the batch
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
      return None # Return None if the whole batch is empty
    # Use the default collate_fn for the rest of the batch
    return torch.utils.data.dataloader.default_collate(batch)

def get_train_loader(train_dataset, parallel, on_gpu, batch_size, n_cpus) -> torch.utils.data.DataLoader:

    if parallel == 1:
        if on_gpu:
            n_threads = torch.cuda.device_count()
        else:
            n_threads = torch.get_num_threads()
        num_workers = n_cpus // n_threads

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
        drop_last=(train_sampler is None)
    )

    return train_dataloader

def get_val_loader(val_dataset, parallel, on_gpu, batch_size, n_cpus) -> torch.utils.data.DataLoader:

    if parallel == 1:
        if on_gpu:
            n_threads = torch.cuda.device_count()
        else:
            n_threads = torch.get_num_threads()
        num_workers = n_cpus // n_threads

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
        sampler=val_sampler,
    )

    return val_dataloader
