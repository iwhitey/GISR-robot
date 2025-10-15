import sys, os
from os.path import exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_kinematics as pk
import numpy as np

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, PointLights,
    PerspectiveCameras
)


from IPython import embed

class UR5MeshRenderer():

    def __init__ (self, robot, mesh_files, device, batch_size, image_size):

        self.batch_size = batch_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.image_size = image_size
        self.preload_mesh_num_verts = []
        preload_verts = []
        preload_faces = []
        
        # preload the mesh only once
        for m_file in mesh_files:
            assert exists(m_file)
            preload_verts_i, preload_faces_idx_i, aux = load_obj(m_file, load_textures=False, create_texture_atlas=False, 
                                                               device=device)
            
            preload_faces_i = preload_faces_idx_i.verts_idx
            preload_verts.append(preload_verts_i)
            preload_faces.append(preload_faces_i)
            self.preload_mesh_num_verts.append(preload_verts_i.shape[0])

        verts_count = 0
        for i, verts in enumerate(preload_verts):
            preload_faces[i] += verts_count
            verts_count += verts.shape[0]

        mesh_preload = Meshes(
            verts=preload_verts,   
            faces=preload_faces,
        )

        self.preload_verts_padded = mesh_preload.verts_padded().repeat(self.batch_size, 1, 1, 1) # (bs, num_meshes, max_verts, 3)
        self.faces_packed = torch.concat(preload_faces, dim=0).repeat(self.batch_size, 1, 1) # (bs, sum(V_n), F)

    def get_batch_robot_mesh(self, joint_angles):
        
        R_batch, t_batch = self.robot.get_T_all_links(joint_angles)
        assert len(self.mesh_files) == R_batch.shape[1] and len(self.mesh_files) == t_batch.shape[1]
        assert R_batch.shape[1] == self.preload_verts_padded.shape[1]

        preload_verts_padded = self.preload_verts_padded.detach()
        verts = preload_verts_padded @ R_batch.permute(0, 1, -1, -2) + t_batch[:, :, None, :]

        verts_0 = verts[:, 0, :self.preload_mesh_num_verts[0], :]
        verts_1 = verts[:, 1, :self.preload_mesh_num_verts[1], :]
        verts_2 = verts[:, 2, :self.preload_mesh_num_verts[2], :]
        verts_3 = verts[:, 3, :self.preload_mesh_num_verts[3], :]
        verts_4 = verts[:, 4, :self.preload_mesh_num_verts[4], :]
        verts_5 = verts[:, 5, :self.preload_mesh_num_verts[5], :]
        verts_6 = verts[:, 6, :self.preload_mesh_num_verts[6], :]

        verts_packed = torch.concat([verts_0, verts_1, verts_2, verts_3, verts_4, verts_5, verts_6], dim=1) # (bs, sum(V_n), 3)
        faces_packed = self.faces_packed.detach()

        robot_mesh = Meshes(
                verts=verts_packed,   
                faces=faces_packed,
        )

        return robot_mesh

        
    
    def render_mask_from_point_cloud(self, joint_angles, R, t):
        
        robot_mesh = self.get_batch_robot_mesh(joint_angles)
        points_3d = sample_points_from_meshes(robot_mesh, num_samples=70000, return_normals=False, return_textures=False)
        points_3d = points_3d @ self.robot.rot_xz
        img_h, img_w = self.image_size
        camera = FoVPerspectiveCameras(device=self.device, R=R, T=t)
        pts_2d = camera.transform_points_screen(points_3d, image_size = self.image_size)[:, :, :2] # use camera's transform method
        # pts_2d = camera.transform_points(points_3d)[:, :, :2]
        
        # This is how we get the 3D keypoints in the camera coordinate frame!
        # points_3d_cam = camera.get_world_to_view_transform().transform_points(points_3d)
        # pts_3d_cam_manual = points_3d @ R + t[:,None,:] # row-wise R
        # assert torch.allclose(points_3d_cam, pts_3d_cam_manual)
        # depth = points_3d_cam[:, :, 2:]
        # xy = pts_2d[:, :, :2]
        # xy_depth = torch.cat((xy, depth), dim=2)
        # xyz_unproj_world = camera.unproject_points(xy_depth, world_coordinates=True)
        # xyz_unproj_cam = camera.unproject_points(xy_depth, world_coordinates=False)
        # print(torch.allclose(points_3d, xyz_unproj_world, atol=1e-6, rtol=1e-6)) # True
        # print(torch.allclose(points_3d_cam, xyz_unproj_cam, atol=1e-6, rtol=1e-6)) # True
        #-----------------------------------------------_#

        mask = torch.zeros((pts_2d.shape[0], img_h, img_w), device=self.device).float()
        # Extract x and y coordinates from pts_2d
        x_coords = pts_2d[:, :, 0].long().clamp(0, img_w-1)#.reshape(pts_2d.shape[0], 1, pts_2d.shape[1])
        y_coords = pts_2d[:, :, 1].long().clamp(0, img_h-1)#.reshape(pts_2d.shape[0], 1, pts_2d.shape[1])
        mask[torch.arange(mask.shape[0])[:, None], y_coords, x_coords] = 1.0 #[:, None, None]
        mask = mask.unsqueeze(1)
        return mask



    def render_from_pytorch3d(self, joint_angles, R, t):
        robot_mesh = self.get_batch_robot_mesh(joint_angles)
        camera = FoVPerspectiveCameras(device=self.device)
        renderer = self.get_renderer(camera)
        rendered_images = renderer(robot_mesh, R=R, T=t)
        rendered_images = rendered_images.permute(0, -1, 1, 2) # H,W,C -> C,H,W
        rendered_rgb = rendered_images[:, :3, ...]
        mask_pytorch = rendered_images[:, 3, ...].unsqueeze(1) # channel dimension is lost after indexing
        return mask_pytorch, rendered_rgb

    def get_renderer(self, camera, device="cpu"):
        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the object is facing the -Z direction. 

        raster_settings = RasterizationSettings(
            image_size=224, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # The front of the object is facing the -z direction.
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=camera,
                lights=lights
            )
        )

        # blend_params_sil = BlendParams(sigma=1e-8, gamma=1e-8)
        # raster_settings_sil = RasterizationSettings(
        #     image_size=224, 
        #     blur_radius=np.log(1. / 1e-4 - 1.) * blend_params_sil.sigma, 
        #     faces_per_pixel=100,
        #     bin_size=None,
        #     max_faces_per_bin=1000000,  # max_faces_per_bin=1000000,  
        # )
        
        # # Silhouette renderer, only renders mask, rgb is all ones
        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=camera, 
        #         raster_settings=raster_settings_sil
        #     ),
        #     shader=SoftSilhouetteShader(blend_params=blend_params_sil)
        # )

        return renderer    