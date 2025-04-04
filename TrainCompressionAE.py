#!/usr/bin/env python

import os
from copy import deepcopy
from CompressionAEModel import VGGAutoEncoder, get_configs
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import ShapeNetDataLoader
from Dataset import generate_datasets
import json
from types import SimpleNamespace

def setup_ddp(parallel):
    if parallel == 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend='nccl')
    else:
        pass

def prepare_training_objects(batch_size, n_cpus, n_epochs, lr, momentum, weight_decay, parallel=1):
    configs = get_configs('vgg16')
    model = VGGAutoEncoder(configs=configs)
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    split_datasets_dict = generate_datasets(dataset_path=params.DATASET_PATH,
                                            portions = {'train': .5, 'val': .1, 'test': .1})

    train_loader = ShapeNetDataLoader.get_train_loader(train_dataset=split_datasets_dict['train'],
                                                       batch_size=batch_size,
                                                       n_cpus=n_cpus,
                                                       parallel=parallel)

    val_loader = ShapeNetDataLoader.get_val_loader(val_dataset=split_datasets_dict['val'],
                                               parallel=parallel,
                                               batch_size=batch_size,
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
                 save_every: int,
                 print_every: int,
                 snapshot_dir: str,
                 snapshot_path: str = None,):
        if parallel:
            self.gpu_id = int(os.environ['LOCAL_RANK'])
            self.device = torch.device(f'cuda:{self.gpu_id}')
            self.model = model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.device = torch.device('cpu')
            self.gpu_id = 0
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
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device)
        model_dict = self.model.state_dict()
        new_state_dict = deepcopy(snapshot['state_dict'])
        for key in model_dict.keys():
            if f'module.{key}' in snapshot['state_dict'].keys():
                new_state_dict[key] = snapshot['state_dict'][f'module.{key}']
                del new_state_dict[f'module.{key}']
        
        print(self.model.load_state_dict(new_state_dict))
        if 'epochs_run' in snapshot.keys():
            self.epochs_run = snapshot["epochs_run"]

        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
        for i_batch, (source, target) in enumerate(self.train_dataloader):
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)

            output = self.model(source)

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

            if i_batch % self.print_every == 0 and self.gpu_id == 0:
                self.writer.add_scalar("Loss/train", loss_sum / (i_batch + 1), epoch * len(self.train_dataloader) + i_batch)
                print(f"Epoch {epoch} | Batch {i_batch} / {len(self.train_dataloader)} | Loss {loss_sum / (i_batch + 1)}")

        self.scheduler.step()

        if epoch % self.save_every == 0 and self.gpu_id == 0:
            self._save_snapshot(epoch)


    def train(self, n_epochs):
        for epoch in range(self.epochs_run, n_epochs):
            if params.PARALLEL:
                self.train_dataloader.sampler.set_epoch(epoch)
            self._run_epoch(epoch)
            self.epochs_run += 1

          # syn for logging
            # torch.cuda.synchronize()
    def validate(self):
        with torch.no_grad():
            self.model.eval()
            loss_sum = 0.0
            for i_batch, source, target in enumerate(self.val_dataloader):
                source = source.to(self.gpu_id)
                target = target.to(self.gpu_id)
                output = self.model(source)

                loss = self.criterion(output, target)
                loss_sum += loss.item()

            self.writer.add_scalar("Loss/val", loss_sum / len(self.val_dataloader), self.epochs_run)


if __name__ == "__main__":
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    if params.PARALLEL:
        setup_ddp(parallel=params.PARALLEL)

    model, optimizer, scheduler, train_dataloader, val_dataloader, criterion = prepare_training_objects(batch_size=int(params.BATCH_SIZE),
                                                                                                        n_cpus=int(params.NUM_WORKERS),
                                                                                                        n_epochs=int(params.N_EPOCHS),
                                                                                                        lr=params.LEARNING_RATE,
                                                                                                        momentum=params.MOMENTUM,
                                                                                                        weight_decay=params.WEIGHT_DECAY,
                                                                                                        parallel=params.PARALLEL)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      criterion=criterion,
                      parallel=params.PARALLEL,
                      save_every=1,
                      print_every=100,
                      snapshot_dir=params.SNAPSHOT_DIR,
                      snapshot_path=os.path.join(params.SNAPSHOT_DIR, 'snapshot.pth'))

    trainer.train(n_epochs=int(params.N_EPOCHS))
    trainer.writer.close()
    destroy_process_group()
