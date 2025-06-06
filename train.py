#!/usr/bin/env python

import os
from copy import deepcopy
from models.builder import build_model
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import ShapeNetDataLoader
from Dataset import generate_datasets, load_dataset
import json
import torch.nn.init as init
from types import SimpleNamespace



def weights_init_orthogonal(module):
    # classname = m.__class__.__name__
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.Linear):
        init.orthogonal_(module.weight.data, gain=1)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)

def setup_ddp(backend, on_gpu):
    local_rank = int(os.environ["LOCAL_RANK"])
    if on_gpu:
        torch.cuda.set_device(local_rank)
    init_process_group(backend=backend)
    # init_process_group(backend='nccl')

    return local_rank

def prepare_training_objects(model_arch, datasets_dict, train_batch_size, val_batch_size, on_gpu, n_cpus, n_epochs, lr, momentum, weight_decay, parallel=1):
    # configs = VGG_AE.get_configs('vgg16')
    # model = VGG_AE.VGGAutoEncoder(configs=configs)
    model = build_model(model_arch)
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    # split_datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH,
    #                                         portions = {'train': .5, 'val': .1, 'test': .1})

    train_loader = ShapeNetDataLoader.get_train_loader(train_dataset=datasets_dict['train'],
                                                       batch_size=train_batch_size,
                                                       n_cpus=n_cpus,
                                                       on_gpu=on_gpu,
                                                       parallel=parallel)

    val_loader = ShapeNetDataLoader.get_val_loader(val_dataset=datasets_dict['val'],
                                                   parallel=parallel,
                                                   on_gpu=on_gpu,
                                                   batch_size=val_batch_size,
                                                   n_cpus=n_cpus)
    criterion = nn.MSELoss()

    return model, optimizer, scheduler, train_loader, val_loader, criterion

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 parallel: bool,
                 on_gpu: bool,
                 save_every: int,
                 print_every: int,
                 snapshot_dir: str,
                 snapshot_path: str = None,):
        if parallel:
            self.node_id = int(os.environ['LOCAL_RANK'])
            if on_gpu:
                self.device = torch.device(f'cuda:{self.node_id}')
                device_ids = [self.node_id]
            else:
                self.device = torch.device('cpu')
                device_ids = None

            self.model = model.to(self.device)
            self.model = DDP(self.model, device_ids=device_ids)
        else:
            self.device = torch.device('cpu')
            self.node_id = 0
            self.model = model

        self.optimizer = optimizer
        self.lr = self.optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.save_every = save_every
        self.print_every = print_every
        self.epochs_run = 0
        self.snapshot_dir = snapshot_dir
        self.writer = SummaryWriter()
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

        if snapshot_path.endswith('.pth') and os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        else:
            self.model.apply(weights_init_orthogonal)

    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device)
        model_dict = self.model.state_dict()
        new_state_dict = deepcopy(snapshot['state_dict'])
        for key in model_dict.keys():
            if 'transition' not in key:
                if f'module.{key}' in snapshot['state_dict'].keys():
                    new_state_dict[key] = snapshot['state_dict'][f'module.{key}']
                    del new_state_dict[f'module.{key}']
        
        print(self.model.load_state_dict(new_state_dict, strict=False))
        if 'epochs_run' in snapshot.keys():
            self.epochs_run = snapshot["epochs_run"]

        print(f"Resuming training from snapshot at Epoch {self.epochs_run+1}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs_run": self.epochs_run
        }
        torch.save(snapshot, os.path.join(self.snapshot_dir, f"snapshot_{epoch}.pth"))

    def _run_epoch(self, epoch):
        self.model.train()
        loss_sum = 0.0
        for i_batch, (source, mask) in enumerate(self.train_dataloader):
            if params.PARALLEL:
                source = source.to(self.device)
                mask = mask.to(self.device)

            output = self.model(source)
            masked_source = mask * source
            masked_output = mask * output
            shape_loss = self.criterion(masked_source,
                                        masked_output)

            background_loss = self.criterion((1-mask) * source,
                                             (1-mask) * output)
            loss = .9 * shape_loss + .1 * background_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.node_id == 0:
                self.writer.add_scalar("Loss/train", loss_sum / (i_batch + 1), epoch * len(self.train_dataloader) + i_batch)
                print(f"TRN Epoch {epoch} | Batch {i_batch} / {len(self.train_dataloader)} | Loss {loss_sum / (i_batch + 1)}")

        self.scheduler.step()

        if epoch % self.save_every == 0 and self.node_id == 0:
            self._save_snapshot(epoch)

    def train(self, n_epochs, do_validate=False):
        for epoch in range(self.epochs_run, n_epochs):
            if params.PARALLEL:
                self.train_dataloader.sampler.set_epoch(epoch)
            self._run_epoch(epoch)
            if do_validate:
                self.validate(epoch)
            self.epochs_run += 1

    def validate(self, epoch: int):
        with torch.no_grad():
            self.model.eval()
            loss_sum = 0.0
            for i_batch, (source, mask) in enumerate(self.val_dataloader):
                source = source.to(self.device)
                mask = mask.to(self.device)

                output = self.model(source)
                masked_source = mask * source
                masked_output = mask * output
                shape_loss = self.criterion(masked_source,
                                            masked_output)

                background_loss = self.criterion((1 - mask) * source,
                                                 (1 - mask) * output)
                loss = .95 * shape_loss + .05 * background_loss
                loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.node_id == 0:
                self.writer.add_scalar("Loss/val", loss_sum / len(self.val_dataloader), self.epochs_run)
                print(f"VAL Epoch {epoch} | Batch {i_batch} / {len(self.val_dataloader)} | Loss {loss_sum / (i_batch + 1)}")



if __name__ == "__main__":
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    if params.GPU:
        assert params.BACKEND == 'nccl'
    else:
        assert params.BACKEND == 'gloo'

    if params.PARALLEL:
        rank = setup_ddp(backend=params.BACKEND, on_gpu=params.GPU)
    else:
        rank = 0

    if rank == 0:
        datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH,
                                          portions = {'train': .1, 'val': .03, 'test': .02},
                                          use_prev_indices=params.USE_PREV_INDICES)
        if params.PARALLEL:
            torch.distributed.barrier()

    else:
        if params.PARALLEL:
            torch.distributed.barrier()
        datasets_dict = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets_dict[split_name] = load_dataset(split_name=split_name)
    print('All nodes in sync, starting training...')
    model_arch = 'vgg11'
    model, optimizer, scheduler, train_dataloader, val_dataloader, criterion = prepare_training_objects(model_arch=model_arch,
                                                                                                        datasets_dict=datasets_dict,
                                                                                                        train_batch_size=int(params.TRAIN_BATCH_SIZE),
                                                                                                        val_batch_size=int(params.VAL_BATCH_SIZE),
                                                                                                        n_cpus=int(params.NUM_WORKERS),
                                                                                                        on_gpu=params.GPU,
                                                                                                        n_epochs=int(params.N_EPOCHS),
                                                                                                        lr=params.LEARNING_RATE,
                                                                                                        momentum=params.MOMENTUM,
                                                                                                        weight_decay=params.WEIGHT_DECAY,
                                                                                                        parallel=params.PARALLEL)
    snapshot_dir = os.path.join(params.SNAPSHOT_DIR, model_arch)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      criterion=criterion,
                      parallel=params.PARALLEL,
                      on_gpu=params.GPU,
                      save_every=1,
                      print_every=1000,
                      snapshot_dir=snapshot_dir,
                      snapshot_path=os.path.join(params.SNAPSHOT_DIR, 'caltech256-vgg11.pth'))

    trainer.train(n_epochs=int(params.N_EPOCHS), do_validate=params.DO_VALIDATE)
    trainer.writer.close()
    destroy_process_group()
