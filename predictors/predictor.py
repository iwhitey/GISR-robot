import torch
import torch.nn as nn

from networks.edm_utils import get_edm_model
from networks.convnext import ConvNext

from kornia.geometry.conversions import convert_points_to_homogeneous
from utils.pose_utils import pose_from_pnp, compute_rotation_matrix_from_ortho6d

from torchvision.utils import save_image
from utils.plots import draw_skeleton


class RobotStatePredictor(nn.Module):
    def __init__(self, cfg, robot, renderer, init_model_path = "", network_config = {"type": "convnext_tiny", "q_out_dim": 6, "in_channels": 2}):
        super(RobotStatePredictor, self).__init__()

        self.robot = robot
        self.renderer = renderer
        self.network_config = network_config
        self.refiner = ConvNext(network_config)

        self.cfg = cfg
        use_gim = cfg.gim_config.use_gim
        self.configuration_from_edm = use_gim and cfg.gim_config.init_robot_configuration
        self.rotation_from_pnp = use_gim and cfg.gim_config.rotation_from_pnp
        self.scale_from_kinematic_params = use_gim and cfg.gim_config.scale_from_kinematic_params
        self.scale_from_pnp = use_gim and cfg.gim_config.scale_from_pnp

        if self.configuration_from_edm:
            if init_model_path != "":
                self.edm_model = get_edm_model(init_model_path)
        
        self.q_slice = slice(None, network_config["q_out_dim"])

    def set_init_model(self, init_path):
        self.edm_model = get_edm_model(init_path)

    def _init_q(self, data, joint_angles):
        if self.configuration_from_edm:
            with torch.no_grad():
                init_q = self.edm_model(data)
                init_q = torch.clamp(init_q, self.robot.joint_limits_min[self.q_slice], self.robot.joint_limits_max[self.q_slice])
        else:
            init_q = joint_angles + torch.normal(mean=0., std=0.8, size=joint_angles.shape, device=joint_angles.device)
        return init_q

    def _init_pose(self, init_q, keypoints_2d, p_ref, K):
        
        init_R = torch.eye(3, dtype=torch.float32, device=init_q.device).unsqueeze(0).repeat(init_q.shape[0], 1, 1)
        init_scale = torch.Tensor([1.0]).repeat(init_q.shape[0], 1).to(init_q.device)
        p_ref_hom = convert_points_to_homogeneous(p_ref)
        t_cam_obj_backproj = torch.linalg.solve(K, p_ref_hom)

        if self.rotation_from_pnp:
            init_R, init_t, pts_3d_obj = pose_from_pnp(keypoints_2d, init_q, K, self.robot)
            if self.scale_from_pnp:
                init_scale = init_t[:, -1].unsqueeze(-1)

        if self.scale_from_kinematic_params:
            p_joint_aux_hom = convert_points_to_homogeneous(keypoints_2d[:, 1])
            p_joint_aux_backproj = torch.linalg.solve(K, p_joint_aux_hom)
            init_scale = self.robot.joint_01_length / ((((p_joint_aux_backproj - t_cam_obj_backproj)**2).sum(axis=-1, keepdim=True))**0.5)
           
        return init_R, init_scale, t_cam_obj_backproj


    def forward(
        self, data_id, img_path, input_mask, 
        joint_angles, pose_gt, img_size_render, 
        n_iter, input_img, edm_2d_upper, 
        keypoints_2d, p_ref, 
        K_orig, K_render, 
        focal_length, principal_point,
    ):
        
        init_q = self._init_q(edm_2d_upper, joint_angles).detach()
        init_R, scale, t_cam_obj_uncalib = self._init_pose(init_q, keypoints_2d, p_ref, K_orig)
        rot_curr = {"iter_0": init_R}
        scale_curr = {"iter_0": scale}
        joint_angles_curr = {"iter_0": init_q}

        R_gt = pose_gt[:, :3, :3] # col-wise
        t_gt = pose_gt[:, :3, 3]

        pts_3d_obj_pred = self.robot.get_joint_3d_locations_from_angles(init_q)
        pts_3d_cam_pose_gt = R_gt @ pts_3d_obj_pred.transpose(-1, -2) + t_gt.unsqueeze(-1)
        pts_3d_cam_t_est = R_gt @ pts_3d_obj_pred.transpose(-1, -2) + (scale_curr["iter_0"] * t_cam_obj_uncalib).unsqueeze(-1) #t_gt[:, -1].unsqueeze(-1)
        pts_3d_cam_rot_est = init_R @ pts_3d_obj_pred.transpose(-1, -2) + t_gt.unsqueeze(-1)
        # pts_3d_obj_gt = self.robot.get_joint_3d_locations_from_angles(joint_angles)
        # pts_3d_cam_gt = R_gt @ pts_3d_obj_gt.transpose(-1, -2) + t_gt.unsqueeze(-1)

        init_pose_loss = (pts_3d_cam_t_est - pts_3d_cam_pose_gt).abs().mean() + (pts_3d_cam_rot_est - pts_3d_cam_pose_gt).abs().mean()
        init_q_loss = (init_q - joint_angles).abs().mean(axis=0, keepdim=True)
            
        losses = []; q_loss = []; pose_loss = []; pose_loss_rot_est = []; pose_loss_t_est = []; add_loss = []

        # save_image(input_img[0], f"masks_input_sam_pred_gt/input_img_{data_id[0]}.png")

        for i in range(n_iter): 

            R_curr = rot_curr[f"iter_{i}"].detach()
            scale = scale_curr[f"iter_{i}"].detach()
            t_curr = scale * t_cam_obj_uncalib
            q_curr = joint_angles_curr[f"iter_{i}"].detach()
           
            #---- RGB rendering ----#
            # cameras = PerspectiveCameras(focal_length = -1 * focal_length,
            #                     principal_point = principal_point,
            #                     R=R_curr.transpose(-1, -2), # row-wise transforms
            #                     T=t_curr,
            #                     device=input_img.device,
            #                     in_ndc=False, image_size = img_size_render) # (height, width)

            # robot_mesh = self.renderer.get_batch_robot_mesh(q_curr)
            # rgb_rendered, _ = self.renderer.render_images(robot_mesh, cameras)
            # imgs = torch.cat([input_img, rgb_rendered], dim=1)
            #-----------------------#

            # if input mask is a 1-dim tensor of dummy zeros, then precomputed masks are not available so he have to render it
            if input_mask.dim() == 1:
                input_mask = self.renderer.render_mask_from_point_cloud(joint_angles, R_gt, t_gt, K_render)
            mask = self.renderer.render_mask_from_point_cloud(q_curr, R_curr, t_curr, K_render)
            imgs = torch.cat([input_mask, mask], dim=1)

            q_update, pose_update = self.refiner(imgs)
            q_updated = q_update + q_curr

            if not self.training:
                q_updated = torch.clamp(q_updated, self.robot.joint_limits_min[self.q_slice], self.robot.joint_limits_max[self.q_slice])

            joint_angles_curr[f"iter_{i+1}"] = q_updated
            dR = compute_rotation_matrix_from_ortho6d(pose_update[:, 0:6])
            dz = pose_update[:, 6]
            R_updated = dR @ R_curr
            scale_updated = scale * dz.unsqueeze(-1)
            rot_curr[f"iter_{i+1}"] = R_updated
            scale_curr[f"iter_{i+1}"] = scale_updated

            pts_3d_obj_pred = self.robot.get_joint_3d_locations_from_angles(q_curr)
            pts_3d_cam_pose_gt = R_gt @ pts_3d_obj_pred.transpose(-1, -2) + t_gt.unsqueeze(-1)
            pts_3d_cam_t_est = R_gt @ pts_3d_obj_pred.transpose(-1, -2) + (scale_updated * t_cam_obj_uncalib).unsqueeze(-1)
            pts_3d_cam_rot_est = R_updated @ pts_3d_obj_pred.transpose(-1, -2) + t_gt.unsqueeze(-1)

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
 
        # draw_skeleton(img_path, data_id, self.robot, R_updated, scale_updated * t_cam_obj_uncalib, K_orig, q_updated, line_width=5)

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
