import sys, os
from os.path import exists

import torch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, PointLights, 
    PerspectiveCameras, TexturesAtlas
)

import numpy as np
import kornia
from utils.camera_utils import project_points_on_image

from IPython import embed


class PandaMeshRenderer():

    def __init__ (self, robot, mesh_files, device, batch_size, image_size, focal_length=None, principal_point=None):
        
        self.batch_size = batch_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.image_size = image_size
        self.preload_mesh_num_verts = []
        preload_verts = []
        preload_faces = []
        textures = []
        
        # preload the mesh only once
        for m_file in mesh_files:
            assert exists(m_file)
            preload_verts_i, preload_faces_idx_i, aux = load_obj(m_file, load_textures=True, create_texture_atlas=True, 
                                                               texture_atlas_size=5, texture_wrap='repeat', device=self.device)
            
            preload_faces_i = preload_faces_idx_i.verts_idx
            preload_verts.append(preload_verts_i)
            preload_faces.append(preload_faces_i)
            textures.append(aux.texture_atlas)
            self.preload_mesh_num_verts.append(preload_verts_i.shape[0])

        # manually transform vertices to packed representation
        verts_count = 0
        for i, verts in enumerate(preload_verts):
            preload_faces[i] += verts_count
            verts_count += verts.shape[0]

        mesh_preload = Meshes(
            verts=preload_verts,   
            faces=preload_faces,
            textures = TexturesAtlas(atlas=textures)
        )

        self.preload_verts_padded = mesh_preload.verts_padded().repeat(self.batch_size, 1, 1, 1) # (bs, num_meshes, max_verts, 3)
        self.faces_packed = torch.concat(preload_faces, dim=0).repeat(self.batch_size, 1, 1) # (bs, sum(V_n), F)
        self.textures_packed = torch.concat(textures, dim=0).repeat(self.batch_size, 1, 1, 1, 1) # (bs, sum(V_n), F, F, 3)

        del mesh_preload
        del preload_faces
        del textures
    
        # set up an initial camera and renderer with given camera parameters
        self.cameras = None
        if focal_length is not None and principal_point is not None:
            self.cameras = PerspectiveCameras(focal_length = focal_length,
                                        principal_point = principal_point,
                                        device=device,
                                        in_ndc=False, image_size = [image_size]) #  (height, width)
        

        self.renderer = self.setup_phong_renderer(image_size)



    def setup_soft_phong_renderer(self, image_size):
        ## Soft Phong renderer, renders rgb + mask
        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        blend_params = BlendParams(sigma=1e-4, gamma=1e-8, background_color=(0, 0, 0))
        raster_settings_soft = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*blend_params.sigma, 
            faces_per_pixel=50, 
            max_faces_per_bin=1000000,
            perspective_correct=False, 
        )
        # Differentiable soft renderer using per vertex RGB colors for texture
        soft_phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=self.device, cameras=self.cameras, lights=lights, blend_params=blend_params)
        )
        return soft_phong_renderer

    
    def setup_phong_renderer(self, image_size):
        # Simple Phong renderer, only needs to render one face per pixel.
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-8)#, background_color=(0, 0, 0))
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000, 
        )
        # We can add a point light in front of the object. 
        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        hard_phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device, cameras=self.cameras, lights=lights, blend_params=None)
        )
        return hard_phong_renderer


    def setup_silhouette_renderer(self, image_size):
        blend_params_sil = BlendParams(sigma=1e-8, gamma=1e-8)
        # blend_params_sil = BlendParams(sigma=1e-4, gamma=1e-8) #background_color=(0, 0, 0))
        raster_settings_sil = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params_sil.sigma, 
            faces_per_pixel=100,
            bin_size=None,
            max_faces_per_bin=100000,  # max_faces_per_bin=1000000,  
        )
        
        # renders mask, rgb is all ones
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None, 
                raster_settings=raster_settings_sil
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params_sil)
        )
        return silhouette_renderer
     
    
    def get_batch_robot_mesh(self, joint_angles):
        
        if joint_angles.shape[1] == 6:
            ee_angle = self.robot.joint_limits_min[-1] + 1e-3 + (self.robot.joint_limits_max[-1] - 1e-3 - (self.robot.joint_limits_min[-1] + 1e-3)) * torch.rand((joint_angles.shape[0], 1), device=self.device)
            joint_angles = torch.cat([joint_angles, ee_angle], dim=1)

        R_batch, t_batch = self.robot.get_T_all_links(joint_angles)
        assert len(self.mesh_files) == R_batch.shape[1] and len(self.mesh_files) == t_batch.shape[1]
        assert R_batch.shape[1] == self.preload_verts_padded.shape[1]

        # (bs, N, max_verts, 3) @ (bs, N, 3, 3) + (bs, N, 1, 3) = (bs, N, max_verts, 3)
        preload_verts_padded = self.preload_verts_padded.detach()#.clone()
        verts = preload_verts_padded @ R_batch.permute(0, 1, -1, -2) + t_batch[:, :, None, :]

        # verts are manually transformed from padded to packed representation, i.e. (bs, N, max_verts, 3) -> (bs, sum(V_n), 3)
        # this can't be achieved with simple reshaping since each tensor gets zero-padded individually, so after reshaping padding is "between" tensors, not at the end
        verts_0 = verts[:, 0, :self.preload_mesh_num_verts[0], :]
        verts_1 = verts[:, 1, :self.preload_mesh_num_verts[1], :]
        verts_2 = verts[:, 2, :self.preload_mesh_num_verts[2], :]
        verts_3 = verts[:, 3, :self.preload_mesh_num_verts[3], :]
        verts_4 = verts[:, 4, :self.preload_mesh_num_verts[4], :]
        verts_5 = verts[:, 5, :self.preload_mesh_num_verts[5], :]
        verts_6 = verts[:, 6, :self.preload_mesh_num_verts[6], :]
        verts_7 = verts[:, 7, :self.preload_mesh_num_verts[7], :]
        verts_8 = verts[:, 8, :self.preload_mesh_num_verts[8], :]
        verts_packed = torch.concat([verts_0, verts_1, verts_2, verts_3, verts_4, verts_5, verts_6, verts_7, verts_8], dim=1) # (bs, sum(V_n), 3)

        faces_packed = self.faces_packed.detach()
        textures_packed = self.textures_packed.detach()
        textures_atlas = TexturesAtlas(atlas=textures_packed)
        robot_mesh = Meshes(
                verts=verts_packed,   
                faces=faces_packed,
                textures = textures_atlas
        )
        
        return robot_mesh
    
    def render_mask_from_point_cloud(self, joint_angles, R, t, K, render_size=[224, 224]):
        if joint_angles.shape[1] == 6:
            # ee_angle = self.robot.joint_limits_min[-1] + 1e-3 + (self.robot.joint_limits_max[-1] - 1e-3 - (self.robot.joint_limits_min[-1] + 1e-3)) * torch.rand((joint_angles.shape[0], 1), device=self.device)
            ee_angle = torch.zeros((joint_angles.shape[0], 1), device=self.device)
            joint_angles = torch.cat([joint_angles, ee_angle], dim=1)

        R_batch, t_batch = self.robot.get_T_all_links(joint_angles)
        assert len(self.mesh_files) == R_batch.shape[1] and len(self.mesh_files) == t_batch.shape[1]
        assert R_batch.shape[1] == self.preload_verts_padded.shape[1]

        # (bs, N, max_verts, 3) @ (bs, N, 3, 3) + (bs, N, 1, 3) = (bs, N, max_verts, 3)
        preload_verts_padded = self.preload_verts_padded.detach()#.clone()
        verts = preload_verts_padded @ R_batch.permute(0, 1, -1, -2) + t_batch[:, :, None, :]

        verts_0 = verts[:, 0, :self.preload_mesh_num_verts[0], :]
        verts_1 = verts[:, 1, :self.preload_mesh_num_verts[1], :]
        verts_2 = verts[:, 2, :self.preload_mesh_num_verts[2], :]
        verts_3 = verts[:, 3, :self.preload_mesh_num_verts[3], :]
        verts_4 = verts[:, 4, :self.preload_mesh_num_verts[4], :]
        verts_5 = verts[:, 5, :self.preload_mesh_num_verts[5], :]
        verts_6 = verts[:, 6, :self.preload_mesh_num_verts[6], :]
        verts_7 = verts[:, 7, :self.preload_mesh_num_verts[7], :]
        verts_8 = verts[:, 8, :self.preload_mesh_num_verts[8], :]
        verts_packed = torch.concat([verts_0, verts_1, verts_2, verts_3, verts_4, verts_5, verts_6, verts_7, verts_8], dim=1) # (bs, sum(V_n), 3)
        faces_packed = self.faces_packed.detach()
        textures_packed = self.textures_packed.detach()

        textures_atlas = TexturesAtlas(atlas=textures_packed)
        robot_mesh = Meshes(
                verts=verts_packed,   
                faces=faces_packed,
                textures = textures_atlas
        )

        img_h, img_w = render_size
        points_3d, textures = sample_points_from_meshes(robot_mesh, num_samples=300000, return_normals=False, return_textures=True)

        points_3d_cam = points_3d @ R.permute(0, -1, -2) + t[:, None, :]
        pts_2d = project_points_on_image(points_3d_cam, K) # (bs, num_samples, 2)
        mask = torch.zeros((pts_2d.shape[0], img_h, img_w), device=self.device).float()
        # Extract x and y coordinates from pts_2d
        x_coords = pts_2d[:, :, 0].long().clamp(0, img_w-1)#.reshape(pts_2d.shape[0], 1, pts_2d.shape[1])
        y_coords = pts_2d[:, :, 1].long().clamp(0, img_h-1)#.reshape(pts_2d.shape[0], 1, pts_2d.shape[1])
        mask[torch.arange(mask.shape[0])[:, None], y_coords, x_coords] = 1.0 #[:, None, None]
        mask = mask.unsqueeze(1).float()
        # from torchvision.utils import save_image
        # save_image(mask, "masks_from_pts.png")
        return mask


    @torch.no_grad()
    def render_images(self, robot_mesh, cameras):
        rendered_images = self.renderer(meshes_world=robot_mesh, cameras=cameras)
        rendered_images = rendered_images.permute(0, -1, 1, 2) # H,W,C -> C,H,W
        rendered_rgb = rendered_images[:, :3, ...]
        mask = rendered_images[:, 3, ...].unsqueeze(1)
        return rendered_rgb, mask
    