import sys, os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

from networks.convnext_utils import WarmupLinearSchedule
from predictors.predictor import RobotStatePredictor
from predictors.predictor_synth import RobotStateSynthPredictor
from dataset.dataset_builder import DatasetFactory
from robot import PandaRobotArm, UR5RobotArm
from renderer.mesh_renderer import PandaMeshRenderer
from renderer.mesh_renderer_synth import UR5MeshRenderer
from networks.edm_utils import set_edm_model_eval
from utils.utils import print_and_log_to_file, seed_torch

from accelerate import Accelerator, DistributedDataParallelKwargs

torch.set_printoptions(precision=3)


def compute_and_log_eval_metrics(log_path, epoch, num_batches, accelerator, *args):
    if accelerator.is_local_main_process:
        losses = [loss / num_batches for loss in args]
        prefix = f"Validation | Epoch {epoch} | "
        q_loss_total, q_loss_per_iter, init_mae_q_total, pose_loss_total, pose_loss_per_iter, pose_loss_rot_est_total, pose_loss_t_est_total, init_pose_loss_total = losses
        log_dict = {prefix+"MAE_q_init = " : f"{init_mae_q_total * 180 / 3.14159}",
                    prefix+"MAE_q_all_iter = " : f"{q_loss_total * 180 / 3.14159}",
                    prefix+"MAE_q_per_iter = " : f"{(q_loss_per_iter * 180 / 3.14159).data}",
                    prefix+"pose_loss_init = " : f"{init_pose_loss_total:.3f}",
                    prefix+"pose_loss_all_iter = " : f"{pose_loss_total:.3f}",
                    prefix+"pose_loss_rot_est_all_iter = " : f"{pose_loss_rot_est_total:.3f}",
                    prefix+"pose_loss_t_est_all_iter = " : f"{pose_loss_t_est_total:.3f}",
                    prefix+"pose_loss_per_iter = " : f"{pose_loss_per_iter.data}"
                    }
        
        print_and_log_to_file(log_path, log_dict)
        

def validate(model, val_dataloader, epoch, n_iter, cfg, accelerator):

    device = accelerator.device
    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Validation")

    q_loss_total = 0; q_loss_per_iter = 0; 
    init_mae_q_total = 0; pose_loss_total = 0; 
    pose_loss_per_iter = 0; pose_loss_rot_est_total = 0; 
    pose_loss_t_est_total = 0; init_pose_loss_total = 0

    model.eval()

    if hasattr(model, 'module'):
        refiner = model.module.refiner
        if hasattr(model.module, 'edm_model'):
            assert not model.module.edm_model.training
    else:
        refiner = model.refiner
        if hasattr(model, 'edm_model'):
            assert not model.edm_model.training
    assert not refiner.training

    with torch.no_grad():

        for idx, data in enumerate(val_dataloader):

            data_id = data['data_id']
            mask = data['mask'].to(device)
            pose_gt = data['pose'].to(device)
            joint_angles = data['joint_angles_gt'].to(device)
            img_size_render = data['img_size_render'].to(device)
            img_path = data['img_path']

            # these are used for panda robot only
            img = data['img'].to(device)
            edm_2d_upper = data['edm_2d_upper'].to(device)
            keypoints_2d = data['keypoints_2d_gt'].to(device)
            p_ref = data['robot_base_2d'].to(device)
            f_xy = data['fx_fy_render'].to(device)
            c_xy = data['cx_cy_render'].to(device)
            K_orig = data['K_orig'].to(device)
            K_render = data['K_render'].to(device)
            
            losses_val_all = model(
                                data_id, img_path, mask, 
                                joint_angles, pose_gt,  
                                img_size_render, n_iter, 
                                input_img=img, edm_2d_upper=edm_2d_upper, 
                                keypoints_2d=keypoints_2d, p_ref=p_ref, 
                                K_orig=K_orig, K_render=K_render, focal_length=f_xy, principal_point=c_xy
                            )
                
            _, _, _, _, q_loss, init_q_loss, pose_loss, pose_loss_rot_est, pose_loss_t_est, init_pose_loss = accelerator.gather(losses_val_all)

            q_loss_total += q_loss.mean()
            q_loss_per_iter += q_loss.mean(axis=-1).mean(axis=0)
            init_mae_q_total += init_q_loss.mean()
            pose_loss_total += pose_loss.mean()
            pose_loss_per_iter += pose_loss.mean(axis=0)
            pose_loss_rot_est_total += pose_loss_rot_est.mean()
            pose_loss_t_est_total += pose_loss_t_est.mean()
            init_pose_loss_total += init_pose_loss.mean()

            progress_bar.update(1)

    
    compute_and_log_eval_metrics(
        cfg.log_path, epoch, 
        len(val_dataloader), 
        accelerator, 
        q_loss_total, 
        q_loss_per_iter, 
        init_mae_q_total, 
        pose_loss_total, 
        pose_loss_per_iter, 
        pose_loss_rot_est_total, 
        pose_loss_t_est_total, 
        init_pose_loss_total
        )


def train_iter(cfg):

    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size
    data_root = cfg.train_data_root
    dataset = cfg.train_dataset
    val_data_root = cfg.val_data_root
    val_dataset = cfg.val_dataset
    
    seed_torch()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs], 
        gradient_accumulation_steps=cfg.grad_accumulation_steps
        )
    device = accelerator.device
    
    train_data = DatasetFactory(cfg.robot, is_train=True, data_root=data_root, dataset=dataset, complete_edm=True, cfg=cfg).create()
    val_data = DatasetFactory(cfg.robot, is_train=True, data_root=val_data_root, dataset=val_dataset, complete_edm=True, cfg=cfg).create()

    if cfg.robot == "panda":
        # focal_length, principal_point, img_size = train_data.get_cam_intrinsics_after_resize_and_crop(cfg)
        robot = PandaRobotArm(urdf_path=cfg.urdf_path, ee_name=cfg.ee_name, device=device)
        img_size = [cfg.img_h_model_resize_and_crop, cfg.img_w_model_resize_and_crop]
        renderer = PandaMeshRenderer(robot, cfg.mesh_files, device, batch_size, img_size)
        model = RobotStatePredictor(cfg, robot, renderer, init_model_path=cfg.gim_config.edm_model_path)

    elif cfg.robot == "ur5":
        robot = UR5RobotArm(urdf_path=cfg.urdf_path, ee_name=cfg.ee_name, device=device)
        img_size = [cfg.img_h_model_resize_and_crop, cfg.img_w_model_resize_and_crop]
        renderer = UR5MeshRenderer(robot, cfg.mesh_files, device, batch_size, img_size)
        model = RobotStateSynthPredictor(cfg, robot, renderer)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=4)
    print("Train dataset size: ", len(train_data))
    print("Validation dataset size: ", len(val_data))

    model.to(device)

    lr = cfg.lr #* accelerator.num_processes
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    if cfg.load_model_checkpoint:
        if cfg.load_ddp_model:
            state_dict = torch.load(cfg.checkpoint_path, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove "module." from key since it was saved from DDP"
                new_state_dict[name] = v
        else:
            new_state_dict = torch.load(cfg.checkpoint_path, map_location='cuda:0')
        
        model.load_state_dict(new_state_dict)
        print(f"Loaded model from {cfg.checkpoint_path}")

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_epochs * len(train_dataloader), t_total=num_epochs * len(train_dataloader))

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, scheduler)
    batches_per_epoch_per_device = len(train_dataloader)
    print(f"batches_per_epoch_per_GPU: {batches_per_epoch_per_device}")

    n_iter = torch.ones(num_epochs).int()
    n_iter[:int(num_epochs * 0.25)] = 1
    n_iter[int(num_epochs * 0.25) : int(num_epochs * 0.5)] = 2
    n_iter[int(num_epochs * 0.5):] = cfg.n_iter

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    for epoch in range(num_epochs):
        
        model.train()
        set_edm_model_eval(model)
        optimizer.zero_grad(set_to_none=True)
        
        n_iter_epoch = n_iter[epoch].item()
        loss_epoch = 0.0; q_loss_epoch = 0.0; q_gain_epoch = 0.0; 
        pose_loss_epoch = 0.0; pose_loss_rot_est_epoch = 0.0; pose_loss_t_est_epoch = 0.0
        
        data_ids = []
        for idx, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                # these are used for both panda and ur5
                data_id = data['data_id']
                mask = data['mask'].to(device)
                pose_gt = data['pose'].to(device)
                joint_angles = data['joint_angles_gt'].to(device)
                img_size_render = data['img_size_render'].to(device)
                img_path = data['img_path']

                # these are only used with panda robot, ignored for ur5
                img = data['img'].to(device)
                edm_2d_upper = data['edm_2d_upper'].to(device)
                keypoints_2d = data['keypoints_2d_gt'].to(device)
                p_ref = data['robot_base_2d'].to(device)
                f_xy = data['fx_fy_render'].to(device)
                c_xy = data['cx_cy_render'].to(device)
                K_orig = data['K_orig'].to(device)
                K_render = data['K_render'].to(device)

                (
                losses,
                q_loss, init_q_loss,
                pose_loss, pose_loss_rot_est, pose_loss_t_est, init_pose_loss
                ) = model(
                    data_id, img_path, mask,
                    joint_angles, pose_gt, img_size_render, n_iter_epoch,
                    input_img=img, edm_2d_upper=edm_2d_upper,
                    keypoints_2d=keypoints_2d, p_ref=p_ref,
                    K_orig=K_orig, K_render=K_render,
                    focal_length=f_xy, principal_point=c_xy
                    )
                
                log_str = (
                    f"Epoch {epoch} | batch {idx+1}/{batches_per_epoch_per_device} " 
                    f"| loss = {losses.item():.4f}\nEpoch {epoch} | mae_q = {q_loss.mean(axis=-1).item():.3f} " 
                    f"| q_gain = {(init_q_loss.mean() - q_loss.mean(axis=-1)).item():.3f}\nEpoch {epoch} "
                    f"| pose_loss = {pose_loss.item():.3f} "
                    f"| pose_loss_rot_est = {pose_loss_rot_est.item():.3f} pose_loss_t_est = {pose_loss_t_est.item():.3f} "
                    f"| init_pose_loss = {init_pose_loss.item():.3f}\n"
                    )

                print_and_log_to_file(cfg.log_path, log_str)

                loss_epoch += losses.detach().mean()
                q_loss_epoch += q_loss.detach().mean()
                q_gain_epoch += (init_q_loss.detach().mean() - q_loss.detach().mean(axis=-1)).mean()
                pose_loss_epoch += pose_loss.detach().mean()
                pose_loss_rot_est_epoch += pose_loss_rot_est.detach().mean()
                pose_loss_t_est_epoch += pose_loss_t_est.detach().mean()

                loss = losses.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm, norm_type=2)

                optimizer.step()
                scheduler.step() # for batch-based schedulers
                optimizer.zero_grad(set_to_none=True)

        log_str = (
            f"Epoch end | loss_avg = {loss_epoch.item() / batches_per_epoch_per_device} "
            f"| mae_q_avg = {q_loss_epoch.item() / batches_per_epoch_per_device} "
            f"| q_gain_avg = {q_gain_epoch.item() / batches_per_epoch_per_device}\n"
            f"| pose_loss_avg = {pose_loss_epoch.item() / batches_per_epoch_per_device} "
            f"| pose_loss_rot_est_avg = {pose_loss_rot_est_epoch.item() / batches_per_epoch_per_device} "
            f"| pose_loss_t_est_avg = {pose_loss_t_est_epoch.item() / batches_per_epoch_per_device}"
        )

        print_and_log_to_file(cfg.log_path, log_str)

        if (epoch+1) % cfg.validation_freq == 0:
            validate(model, val_dataloader, epoch, n_iter_epoch, cfg, accelerator)

        if (epoch+1) % cfg.save_model_freq == 0:
            torch.save(model.state_dict(), f"{os.path.join(cfg.save_model_path, f'model_{epoch+1}.pth')}")


    end.record()
    torch.cuda.synchronize()
    log_str = f"Training time (s): {start.elapsed_time(end) / 1000}"
    print_and_log_to_file(cfg.log_path, log_str)



