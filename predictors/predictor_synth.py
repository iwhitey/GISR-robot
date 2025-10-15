import torch
import torch.nn as nn

from networks.convnext import ConvNext

from utils.pose_utils import compute_rotation_matrix_from_ortho6d, random_look_at_pose

from torchvision.utils import save_image


class RobotStateSynthPredictor(nn.Module):
    def __init__(self, cfg, robot, renderer, network_config = {"type": "convnext_tiny", "q_out_dim": 6, "in_channels": 2}):
        super(RobotStateSynthPredictor, self).__init__()

        self.robot = robot
        self.renderer = renderer
        self.cfg = cfg
        self.network_config = network_config
        self.refiner = ConvNext(network_config)
        self.device = cfg.device
        
    def _init_q(self, joint_angles):
        q = joint_angles + torch.normal(mean=0., std=0.5, size=joint_angles.shape, device=self.device)
        return q

    def _init_pose(self, R, t):
        R_delta, t_delta = random_look_at_pose(R.shape[0], interval={"dist":[-0.5, 0.5],"elev":[-20, 20],"azim":[-30, 30]}, device=self.device)
        R = R @ R_delta
        scale = t[:, -1].unsqueeze(-1) + t_delta[:, -1].unsqueeze(-1)
        t_cam_obj_uncalib = t / t[:, -1].unsqueeze(-1) # we assume 2D robot reference keypoint is detected and cam. intrinsic matrix is known
        return R, scale, t_cam_obj_uncalib

    def forward(
        self, data_id, img_path, input_mask, 
        joint_angles, pose_gt, img_size_render, 
        n_iter,
        **kwargs
    ):
        
        R_gt = pose_gt[:, :3, :3] # row-wise
        t_gt = pose_gt[:, 3, :3]
    
        init_q = self._init_q(joint_angles)
        init_R, init_scale, t_cam_obj_uncalib = self._init_pose(R_gt, t_gt)
        rot_curr = {"iter_0": init_R}
        scale_curr = {"iter_0": init_scale}
        joint_angles_curr = {"iter_0": init_q}

        pts_3d_obj_pred = self.robot.get_joint_3d_locations_from_angles(init_q)
        pts_3d_cam_pose_gt = pts_3d_obj_pred @ R_gt + t_gt[:, None, :]
        pts_3d_cam_t_est = pts_3d_obj_pred @ R_gt + (init_scale * t_cam_obj_uncalib)[:, None, :]
        pts_3d_cam_rot_est = pts_3d_obj_pred @ init_R + t_gt[:, None, :]
        init_pose_loss = (pts_3d_cam_t_est - pts_3d_cam_pose_gt).abs().mean() + (pts_3d_cam_rot_est - pts_3d_cam_pose_gt).abs().mean()
        init_q_loss = (init_q - joint_angles).abs().mean(axis=0, keepdim=True)
            
        losses = []; q_loss = []; pose_loss = []; pose_loss_rot_est = []; pose_loss_t_est = []; add_loss = []

        for i in range(n_iter): 

            R_curr = rot_curr[f"iter_{i}"].detach()
            scale = scale_curr[f"iter_{i}"].detach()
            t_curr = scale * t_cam_obj_uncalib
            q_curr = joint_angles_curr[f"iter_{i}"].detach()

            mask = self.renderer.render_mask_from_point_cloud(q_curr, R_curr, t_curr)
            ref_mask = self.renderer.render_mask_from_point_cloud(joint_angles, R_gt, t_gt)

            # if i == n_iter-1:
            #     save_image(mask[0], f"check_train_masks/{data_id[0]}_pred_mask.jpg")
            #     save_image(ref_mask[0], f"check_train_masks/{data_id[0]}_input_mask.png")

            imgs = torch.cat([ref_mask, mask], dim=1)

            q_update, pose_update = self.refiner(imgs)
            q_updated = q_update + q_curr
            joint_angles_curr[f"iter_{i+1}"] = q_updated
            dR = compute_rotation_matrix_from_ortho6d(pose_update[:, 0:6])
            dz = pose_update[:, 6]
            R_updated = R_curr @ dR
            scale_updated = scale * dz.unsqueeze(-1)
            rot_curr[f"iter_{i+1}"] = R_updated
            scale_curr[f"iter_{i+1}"] = scale_updated

            pts_3d_obj_pred = self.robot.get_joint_3d_locations_from_angles(q_curr)
            pts_3d_cam_pose_gt = pts_3d_obj_pred @ R_gt + t_gt[:, None, :]
            pts_3d_cam_t_est = pts_3d_obj_pred @ R_gt + (scale_updated * t_cam_obj_uncalib)[:, None, :]
            pts_3d_cam_rot_est = pts_3d_obj_pred @ R_updated + t_gt[:, None, :]

            pose_rot_est_loss = (pts_3d_cam_rot_est - pts_3d_cam_pose_gt).abs().mean()
            pose_t_est_loss = (pts_3d_cam_t_est - pts_3d_cam_pose_gt).abs().mean()
            pose_loss_i = pose_rot_est_loss + pose_t_est_loss

            pose_loss.append(pose_loss_i.unsqueeze(0))
            pose_loss_rot_est.append(pose_rot_est_loss.unsqueeze(0))
            pose_loss_t_est.append(pose_t_est_loss.unsqueeze(0))

            q_loss_i = (q_updated - joint_angles).abs().mean(axis=0, keepdim=True)
            q_loss.append(q_loss_i)

            q_pred = torch.cat((torch.sin(q_updated), torch.cos(q_updated)), dim=-1)
            q_goal = torch.cat((torch.sin(joint_angles), torch.cos(joint_angles)), dim=-1)
            loss_sc = torch.mean(torch.abs(q_pred - q_goal))

            loss_iter_i = loss_sc + pose_loss_i
            losses.append(loss_iter_i.unsqueeze(0))
        

        losses = torch.cat(losses, dim=0).unsqueeze(0) # (1, n_iter)
        pose_loss = torch.cat(pose_loss, dim=0).unsqueeze(0) # (1, n_iter)
        pose_loss_rot_est = torch.cat(pose_loss_rot_est, dim=0).unsqueeze(0) # (1, n_iter)
        pose_loss_t_est = torch.cat(pose_loss_t_est, dim=0).unsqueeze(0) # (1, n_iter)
        q_loss = torch.cat(q_loss, dim=0).unsqueeze(0) # (1, n_iter, n_joints)

        if self.training:
            return (
                losses, 
                q_loss, init_q_loss, 
                pose_loss, pose_loss_rot_est, pose_loss_t_est, init_pose_loss
            )
        else:
            return (
                q_updated, 
                R_updated, scale_updated * t_cam_obj_uncalib, 
                losses, 
                q_loss, init_q_loss, 
                pose_loss, pose_loss_rot_est, pose_loss_t_est, init_pose_loss
            )


