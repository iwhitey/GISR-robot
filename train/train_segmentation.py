import sys, os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

from networks.convnext_utils import WarmupLinearSchedule
from networks.segmentation import RobotSegNet
from dataset.dataset import SegmentPandaDataset
from utils.utils import print_and_log_to_file, seed_torch

from accelerate import Accelerator, DistributedDataParallelKwargs

torch.set_printoptions(precision=3)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


@accelerator.on_local_main_process
def log_eval_metrics(mae_soft_avg, mae_hard_avg, epoch, log_path):
    log_str = f"Validation: Epoch {epoch} | MAE_soft: {mae_soft_avg} | MAE_hard: {mae_hard_avg}"
    print_and_log_to_file(log_path, log_str)


def validate(model, val_dataloader, epoch, cfg):

    device = accelerator.device
    model.eval()

    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Validation")

    mae_soft_avg = 0.0; mae_hard_avg = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            data_id = data['data_id']
            img = data['img'].to(device)
            mask = data['mask'].to(device)
            mask_logits = model(img)
            mask_logits, mask = accelerator.gather([mask_logits, mask])
            mask_pred = torch.sigmoid(mask_logits).detach()
            mask_pred_hard = (mask_pred > 0.5).float()
            mae_soft = torch.mean(torch.abs(mask_pred - mask))
            mae_hard = torch.mean(torch.abs(mask_pred_hard - mask))
            mae_soft_avg += mae_soft.item()
            mae_hard_avg += mae_hard.item()

            progress_bar.update(1)

    mae_soft_avg = mae_soft_avg / len(val_dataloader)
    mae_hard_avg = mae_hard_avg / len(val_dataloader)
    log_eval_metrics(mae_soft_avg, mae_hard_avg, epoch, cfg.log_path)


def train_seg(cfg):

    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    data_root = cfg.data_root
    dataset = cfg.dataset
    val_data_root = cfg.val_data_root
    val_dataset = cfg.val_dataset
    
    seed_torch()

    device = accelerator.device

    if cfg.info != '':
        log_path = os.path.join(cfg.save_model_path, "train_info.txt")
        print_and_log_to_file(log_path, cfg.info)

    if isinstance(dataset, list):
        train_datasets = []
        for dataset_i in dataset:
            train_datasets.append(SegmentPandaDataset(data_root, dataset_i, cfg=cfg))
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    else: 
        if "synth" not in dataset:
            train_dataset = SegmentPandaDataset(data_root, dataset, cfg=cfg)
        else:
            train_dataset = SegmentPandaDataset(data_root, dataset, cfg=cfg)
    
    if isinstance(val_dataset, list):
        val_datasets = []
        for dataset_i in val_dataset:
            val_datasets.append(SegmentPandaDataset(val_data_root, dataset_i, cfg=cfg))
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    else:
        val_dataset = SegmentPandaDataset(val_data_root, cfg.val_dataset, cfg=cfg)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))


    lr = 1e-4
    model = RobotSegNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_epochs * len(train_dataloader), t_total=num_epochs * len(train_dataloader))

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, scheduler)
    batches_per_epoch_per_device = len(train_dataloader); print(f"batches_per_epoch_per_GPU: {batches_per_epoch_per_device}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    criterionMSE_sum = torch.nn.MSELoss(reduction='sum')
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    criterions = {"mse_sum": criterionMSE_sum, "bce": criterionBCE}
    
    for epoch in range(num_epochs):     
        optimizer.zero_grad(set_to_none=True)
        model.train()
                
        for idx, data in enumerate(train_dataloader):
            data_id = data['data_id']
            img = data['img'].to(device)
            mask = data['mask'].to(device)
            mask_pred = model(img)

            loss = criterions["bce"](mask_pred, mask.detach())

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm, norm_type=2)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            log_str = f"Epoch {epoch} | batch {idx+1}/{batches_per_epoch_per_device} | bce_loss {loss.item()}"
            print_and_log_to_file(cfg.log_path, log_str)


        if (epoch+1) % cfg.validation_freq == 0:
            validate(model, val_dataloader, epoch, cfg)

        if (epoch+1) % cfg.save_model_freq == 0:
            torch.save(model.state_dict(), f"{os.path.join(cfg.save_model_path, f'model_{epoch+1}.pth')}")


    end.record()
    torch.cuda.synchronize()
    log_str = f"Training time (s): {start.elapsed_time(end) / 1000}"
       