import torch
from torch.utils.data import DataLoader

from networks.segmentation import RobotSegNet
from predictors.predictor import RobotStatePredictor

import numpy as np
from tqdm import tqdm

from dataset.dataset_builder import DatasetFactory
from robot import PandaRobotArm
from renderer.mesh_renderer import PandaMeshRenderer
from utils.utils import print_and_log_to_file, seed_torch
from torchvision.utils import save_image

from collections import OrderedDict

def evaluate(cfg):

    device = cfg.device
    batch_size = cfg.batch_size
    data_root = cfg.data_root
    dataset = cfg.dataset

    seed_torch()
    torch.set_printoptions(precision=3)

    dataset = DatasetFactory("panda", is_train=False, data_root=data_root, dataset=dataset, complete_edm=True, cfg=cfg).create()
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=4)
    print("Test dataset: ", dataset)
    print("Test dataset size: ", len(dataset))

    robot = PandaRobotArm(urdf_path=cfg.urdf_path, ee_name=cfg.ee_name, device=device)
    img_size = [cfg.img_h_model_resize_and_crop, cfg.img_w_model_resize_and_crop]
    renderer = PandaMeshRenderer(robot, cfg.mesh_files, device, batch_size, img_size)

    model = RobotStatePredictor(cfg, robot, renderer)
    if cfg.load_ddp_model:
        state_dict = torch.load(cfg.checkpoint_path, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "edm_model" in k: continue # load edm model separately so we can use a different one if needed
            name = k[7:] # remove "module." from key since it was saved from DDP"
            new_state_dict[name] = v
    else:
        new_state_dict = torch.load(cfg.checkpoint_path, map_location='cuda:0')
   
    model.load_state_dict(new_state_dict)
    print(f"Loaded model from {cfg.checkpoint_path}")

    if cfg.gim_config.use_gim and cfg.gim_config.init_robot_configuration:
        model.set_init_model(cfg.gim_config.edm_model_path)

    model.to(device)
    model.eval()

    # for evaluation we use a deeplabv3 segmentation model, trained as in train_segmentation.py
    if cfg.use_segmentation_model:
        seg_model = RobotSegNet(cfg)
        state_dict_seg = torch.load(cfg.seg_checkpoint_path, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict_seg.items():
            name = k[7:] # remove "module." from key since it was saved from DDP"
            new_state_dict[name] = v
        seg_model.load_state_dict(new_state_dict)
        seg_model = seg_model.to(device)
        seg_model.eval()

    total_time = 0
    q_loss_total = 0; q_loss_per_iter = 0; init_q_loss_total = 0
    pose_loss_total = 0; pose_loss_per_iter = 0; pose_loss_rot_est_total = 0; pose_loss_t_est_total = 0; init_pose_loss_total = 0

    n_batches = len(test_dataloader)
    
    with torch.no_grad():

        add_pred_state = []
        add_pred_state_ee = []
        add_pred_pose = []
        add_pred_q = []

        for idx, data in enumerate(test_dataloader):
            
            # mask = data['mask'].to(device)
            img = data['img'].to(device)
            img_deeplab = data['img_deeplab'].to(device)
            pose_gt = data['pose'].to(device)
            R = pose_gt[:, :3, :3].to(device)
            t = pose_gt[:, :3, 3].to(device)
            joint_angles = data['joint_angles_gt'].to(device)
            edm_2d_upper = data['edm_2d_upper'].to(device)
            keypoints_2d = data['keypoints_2d_gt'].to(device)
            p_ref = data['robot_base_2d'].to(device)
            focal_length = data['fx_fy_render']
            principal_point = data['cx_cy_render']
            img_size_render = data['img_size_render']
            data_id = data['data_id']
            K_orig = data['K_orig'].to(device)
            K_render = data['K_render'].to(device)
            img_path = data['img_path']
            # input_img = data['input_img']

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            mask = 0
            if cfg.use_segmentation_model:
                mask_logits = seg_model(img_deeplab)
                mask_pred = torch.sigmoid(mask_logits).detach()
                mask_pred_hard = (mask_pred > 0.5).float()
                mask = mask_pred_hard

            (
            q_updated,
            R_updated, t_updated,
            losses,
            q_loss, init_q_loss,
            pose_loss, pose_loss_rot_est, pose_loss_t_est, init_pose_loss
            ) = model(
                data_id, img_path, mask,
                joint_angles, pose_gt, img_size_render, cfg.n_iter,
                input_img=img, edm_2d_upper=edm_2d_upper,
                keypoints_2d=keypoints_2d, p_ref=p_ref,
                K_orig=K_orig, K_render=K_render,
                focal_length=focal_length, principal_point=principal_point
            )
                       
            end.record()
            torch.cuda.synchronize()
            total_time = total_time + (start.elapsed_time(end) / 1000)

            pts_3d_obj_pred = robot.get_joint_3d_locations_from_angles(q_updated)
            pts_3d_obj_gt = robot.get_joint_3d_locations_from_angles(joint_angles)
            pts_3d_cam_pred_q = R @ pts_3d_obj_pred.transpose(-1, -2) + t.unsqueeze(-1)
            pts_3d_cam_pred_pose = R_updated @ pts_3d_obj_gt.transpose(-1, -2) + t_updated.unsqueeze(-1)
            pts_3d_cam_pred_state = R_updated @ pts_3d_obj_pred.transpose(-1, -2) + t_updated.unsqueeze(-1)
            pts_3d_cam_gt = R @ pts_3d_obj_gt.transpose(-1, -2) + t.unsqueeze(-1)

            add_pred_state.append(pts_3d_cam_gt.transpose(-1, -2) - pts_3d_cam_pred_state.transpose(-1, -2))
            add_pred_state_ee.append((pts_3d_cam_gt.transpose(-1, -2) - pts_3d_cam_pred_state.transpose(-1, -2))[:, -1, None, :])
            add_pred_pose.append(pts_3d_cam_gt.transpose(-1, -2) - pts_3d_cam_pred_pose.transpose(-1, -2))
            add_pred_q.append(pts_3d_cam_gt.transpose(-1, -2) - pts_3d_cam_pred_q.transpose(-1, -2))

            log_str = (
                f"batch {idx+1}/{n_batches}\n"
                f"loss = {losses.mean(axis=1).data}\n"
                f"mae_q = {q_loss.mean(axis=-1).data}\n"
                f"q_gain = {(init_q_loss.mean() - q_loss.mean(axis=-1)).data}\n"
                f"pose_loss = {pose_loss.data} | pose_loss_rot_est = {pose_loss_rot_est.data} | pose_loss_t_est = {pose_loss_t_est.data} | init_pose_loss = {init_pose_loss.data:.3f}\n"
            )
            print_and_log_to_file(cfg.log_path, log_str)

            q_loss_total += q_loss.mean()
            q_loss_per_iter += q_loss.mean(axis=-1)
            init_q_loss_total += init_q_loss.mean()
            pose_loss_total += pose_loss.mean()
            pose_loss_per_iter += pose_loss.mean(axis=0)
            pose_loss_rot_est_total += pose_loss_rot_est.mean()
            pose_loss_t_est_total += pose_loss_t_est.mean()
            init_pose_loss_total += init_pose_loss.mean()

    q_loss_total = q_loss_total / n_batches
    q_loss_per_iter = q_loss_per_iter / n_batches
    init_q_loss_total = init_q_loss_total / n_batches
    pose_loss_total = pose_loss_total / n_batches
    pose_loss_per_iter = pose_loss_per_iter / n_batches
    pose_loss_rot_est_total = pose_loss_rot_est_total / n_batches
    pose_loss_t_est_total = pose_loss_t_est_total / n_batches
    init_pose_loss_total = init_pose_loss_total / n_batches

    add_pred_state_ee_tensor = torch.cat(add_pred_state_ee, dim=0)
    add_pred_state_tensor = torch.cat(add_pred_state, dim=0)
    add_pred_q_tensor = torch.cat(add_pred_q, dim=0)
    add_pred_pose_tensor = torch.cat(add_pred_pose, dim=0)

    add_pred_state_ee_tensor = torch.norm(add_pred_state_ee_tensor, dim=-1, p=2).mean(axis=-1).cpu().numpy()
    add_pred_state_tensor = torch.norm(add_pred_state_tensor, dim=-1, p=2).mean(axis=-1).cpu().numpy()
    add_pred_q_tensor = torch.norm(add_pred_q_tensor, dim=-1, p=2).mean(axis=-1).cpu().numpy()
    add_pred_pose_tensor = torch.norm(add_pred_pose_tensor, dim=-1, p=2).mean(axis=-1).cpu().numpy()

    auc_threshold = 0.1
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
    counts_state = []; counts_q = []; counts_pose = []; counts_ee = []
    for value in add_threshold_values:
        under_threshold_pred_state = (add_pred_state_tensor <= value).mean()
        under_threshold_pred_state_ee = (add_pred_state_ee_tensor <= value).mean()
        under_threshold_pred_q = (add_pred_q_tensor <= value).mean()
        under_threshold_pred_pose = (add_pred_pose_tensor <= value).mean()

        counts_state.append(under_threshold_pred_state)
        counts_ee.append(under_threshold_pred_state_ee)
        counts_q.append(under_threshold_pred_q)
        counts_pose.append(under_threshold_pred_pose)

    auc_state = np.trapz(counts_state, dx=delta_threshold) / auc_threshold
    auc_ee = np.trapz(counts_ee, dx=delta_threshold) / auc_threshold
    auc_q = np.trapz(counts_q, dx=delta_threshold) / auc_threshold
    auc_pose = np.trapz(counts_pose, dx=delta_threshold) / auc_threshold


    avg_fps = (len(test_dataloader) * batch_size) / total_time
    log_str = f"Runtime: {avg_fps} FPS"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"MAE_q_init = {init_q_loss_total} ({init_q_loss_total * 180 / 3.14159})"
    print_and_log_to_file(cfg.log_path, log_str)
    # log_str = f"MAE_q_all_iter = {q_loss_total} ({q_loss_total * 180 / 3.14159})"
    # print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"MAE_q_per_iter = {q_loss_per_iter.data} ({(q_loss_per_iter * 180 / 3.14159).data})"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"pose_loss_init = {init_pose_loss_total:.3f}"
    print_and_log_to_file(cfg.log_path, log_str)
    # log_str = f"pose_loss_all_iter = {pose_loss_total:.3f}"
    # print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"pose_loss_rot_est_all_iter = {pose_loss_rot_est_total:.3f}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"pose_loss_t_est_all_iter = {pose_loss_t_est_total:.3f}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"pose_loss_per_iter = {pose_loss_per_iter.data}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"AUC(ADD)_state = {auc_state}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"AUC(ADD)_q = {auc_q}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"AUC(ADD)_pose = {auc_pose}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"ADD_state = {add_pred_state_tensor.mean()}"
    print_and_log_to_file(cfg.log_path, log_str)
    log_str = f"ee_ADD_state = {add_pred_state_ee_tensor.mean()}"
    print_and_log_to_file(cfg.log_path, log_str)