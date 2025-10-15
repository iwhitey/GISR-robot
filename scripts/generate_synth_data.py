import sys, os
import json
import argparse

import torch
from pytorch3d.renderer import (
    look_at_view_transform,
)

from torchvision.utils import save_image
from robot import UR5RobotArm, PandaRobotArm
from renderer.mesh_renderer_synth import UR5MeshRenderer
from renderer.mesh_renderer import PandaMeshRenderer

from utils.utils import seed_torch
import numpy as np

sys.path.append('../')

torch.manual_seed(0)

def generate_synth_dataset(dataset_path, num_poses, num_q_per_pose, renderer, robot_arm, device):

    min_elev = -20; max_elev = 20
    min_azim = -30; max_azim = 30
    min_dist = 0.8; max_dist = 1.6
    rotations = []; translations = []
    elevs = []; azims = []

    while(len(rotations) < num_poses):
        elev = (max_elev - min_elev) * torch.rand(1) + min_elev
        azim = (max_azim - min_azim) * torch.rand(1) + min_azim
        dist = (max_dist - min_dist) * torch.rand(1) + min_dist
        R, t = look_at_view_transform(dist, elev, azim, at=((0, 0.5, 0.7)), device=device) # y=0.5 pulls it down, z=0.7 moves it away
        # R, t = look_at_view_transform(1.0, 0, 0, at=(0.25, 0.5, 1.5), device=device)
        if len(R.shape) == 3:
            R = R[0]; t = t[0]

        rotations.append(R); translations.append(t)
        elevs.append(elev); azims.append(azim)

    assert len(rotations) == len(translations)

    num_samples = num_poses * num_q_per_pose

    for i in range(len(rotations)):
        R = rotations[i]
        t = translations[i]

        q = (robot_arm.joint_limits_max - robot_arm.joint_limits_min) * torch.rand((num_q_per_pose, 6), device=device) + robot_arm.joint_limits_min

        R = R.unsqueeze(0).repeat(num_q_per_pose, 1, 1)
        t = t.unsqueeze(0).repeat(num_q_per_pose, 1)
        masks = renderer.render_mask_from_point_cloud(q, R, t)
        # masks, rgb = renderer.render_from_pytorch3d(q, R, t)

        pts_3d_cam = renderer.robot.get_joint_3d_locations_wrt_camera(q, R, t)

        for j in range(num_q_per_pose):
            # rgb_name = f"{i*num_q_per_pose + j:06}_rgb_elev_{elevs[i]}_azim_{azims[i]}.jpg"
            # save_image(rgb[j], os.path.join(dataset_path, rgb_name))
            # img_name = f"{i*num_q_per_pose + j:06}_elev_{elevs[i].item()}_azim_{azims[i].item()}.jpg"
            img_name = f"{i*num_q_per_pose + j:06}.jpg"
            save_image(masks[j], os.path.join(dataset_path, img_name))
            pose_j = np.vstack((R[j].cpu().numpy(), t[j].cpu().numpy()[np.newaxis, :])) # row-wise pose
            np.savetxt(os.path.join(dataset_path, f"{i*num_q_per_pose + j:06}_pose.txt"), pose_j) # load with np.loadtxt('file_name.txt')
            np.savetxt(os.path.join(dataset_path, f"{i*num_q_per_pose + j:06}_q.txt"), q[j].cpu().numpy()[np.newaxis, :])
            np.savetxt(os.path.join(dataset_path, f"{i*num_q_per_pose + j:06}_pts_3d_cam.txt"), pts_3d_cam[j].cpu().numpy())    


def setup_robot_renderer(robot, img_size, root, batch_size, device):
    if robot == "ur5":
        urdf_path = os.path.join(root, "urdfs/ur_description/urdf/universalUR5.urdf")
        robot = UR5RobotArm(urdf_path, ee_name="wrist_3_link", device=device)
        mesh_base_dir = os.path.join(root, "urdfs/ur_description/meshes/ur5/visual")
        mesh_names = ['base.obj', 'shoulder.obj', 'upperarm.obj', 'forearm.obj', 'wrist1.obj', 'wrist2.obj', 'wrist3.obj']
        mesh_files = [os.path.join(mesh_base_dir, mesh_file_name) for mesh_file_name in mesh_names]
        renderer = UR5MeshRenderer(robot, mesh_files, device, batch_size=batch_size, image_size=img_size)
    elif robot == "panda":
        urdf_path = os.path.join(root, "urdfs/panda_description/urdf/panda.urdf")
        robot = PandaRobotArm(urdf_path, ee_name="panda_link8", device=device)
        mesh_base_dir = os.path.join(root, "urdfs/panda_description/meshes")
        mesh_names = ['panda_link0.obj', 'panda_link1.obj', 'panda_link2.obj', 'panda_link3.obj', 'panda_link4.obj', 'panda_link5.obj', 'panda_link6.obj', 'panda_link7.obj', 'panda_link8.obj', 'panda_hand.obj']
        mesh_files = [os.path.join(mesh_base_dir, mesh_file_name) for mesh_file_name in mesh_names]
        renderer = PandaMeshRenderer(robot, mesh_files, device, batch_size=batch_size, image_size=img_size)

    return renderer, robot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, choices=["ur5", "panda"])
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset, a folder will be created.")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--replace_existing", action='store_true', help="Existing folder will be deleted and new dataset will be created.")
    parser.add_argument("--num_poses", type=int, default=10)
    parser.add_argument("--num_q_per_pose", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(root, args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created directory: {dataset_path}")
    else:
        if args.replace_existing:
            import shutil
            shutil.rmtree(dataset_path)
            os.makedirs(dataset_path)
            print(f"Deleted existing directory: {dataset_path}, new one is created")
        else:
            print(f"Directory already exists: {dataset_path}, and replace_existing is not set. Exiting...")

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    renderer, robot_arm = setup_robot_renderer(args.robot, args.img_size, root, batch_size=args.num_q_per_pose, device=device)
    
    generate_synth_dataset(dataset_path, args.num_poses, args.num_q_per_pose, renderer, robot_arm, device)
    print("Synthetic dataset generated successfully")