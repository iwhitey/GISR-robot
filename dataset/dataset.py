import os, sys, json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.models.convnext import ConvNeXt_Tiny_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# from utils import generate_edm
# from img_utils import crop_and_resize

from utils.camera_utils import get_camera_intrinsics, compute_intrinsics_for_render_after_resize_and_crop, intrisic_matrix_from_parameters
from utils.pose_utils import DREAM_Real_Poses

from robot import PandaRobotArm

try:
    import graphik
except:
    sys.path.append("../graphIK")

try:
    import liegroups
except:
    sys.path.append("../liegroups")
try:
    import urdfpy
except:
    sys.path.append("../urdfpy")

import graphik
from graphik.utils.roboturdf import RobotURDF, load_panda_truncated
from graphik.utils.utils import list_to_variable_dict
from graphik.utils.dgp import distance_matrix_from_graph


class EDMDatasetReal(Dataset):
    def __init__(self, root, dataset, complete_edm, robot_type, cfg):
        super(EDMDatasetReal, self).__init__()
        self.root = root
        self.dataset = dataset
        self.dataset_path = os.path.join(self.root, self.dataset)
        self.masks_dir = os.path.join(self.root, dataset+"_gt_masks")
        self.complete_edm = complete_edm
        self.robot_type = robot_type
        self.frames = sorted(os.listdir(self.dataset_path))
        self.frames = [frame for frame in self.frames if ".json" in frame]
        self.camera_file = self.frames[-1]
        self.frames = self.frames[:-1]

        self.num_keypoints_per_image = 7
        self.num_distances_upper = (self.num_keypoints_per_image * (self.num_keypoints_per_image -1)) // 2

        self.is_panda = (self.robot_type == "panda")
        self.data_key = "panda_joint"

        self.robot, self.graph = load_panda_truncated(limits=None)

        self.transform = transforms.Compose([
                ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
        ])

        self.deeplab_transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

        self.K = get_camera_intrinsics(os.path.join(self.dataset_path, self.camera_file))
        self.cfg = cfg

    def __getitem__(self, index):
        img_json = self.frames[index]
        data_id = img_json.split('.')[0]
        img_path = os.path.join(self.dataset_path, data_id + ".rgb.jpg")
        mask_path = os.path.join(self.masks_dir, data_id + "_gt_mask.jpg")

        with open(os.path.join(self.dataset_path, img_json)) as f:
            data = json.load(f)

        keypoints = data['objects'][0]['keypoints']
        joint_data = data["sim_state"]["joints"]
        joint_angles = np.array([joint_data[i]['position'] for i in range(len(joint_data)) if self.data_key in joint_data[i]['name']], dtype=np.float32)[:6]
        
        keypoints_dict = [keypoints[i] for i in range(len(keypoints))]
        keypoints_sorted = sorted(keypoints_dict, key=lambda d: d['name'])
        keypoints_2d = [keypoints_sorted[i]['projected_location'] for i in range(len(keypoints_sorted))]
        keypoints_3d = [keypoints_sorted[i]['location'] for i in range(len(keypoints_sorted))]
        robot_base_2d = np.array(keypoints_2d[1], dtype=np.float32) # 0 is panda hand
        robot_base_3d_cam = np.array(keypoints_3d[1], dtype=np.float32) # 0 is panda hand
        
        keypoints_2d, keypoints_3d = self.adjust_robot_definition(keypoints_2d, keypoints_3d)
        keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32)

        edm_2d = self.compute_edm(keypoints_2d)
        edm_2d_upper = self.compute_edm_upper_triu(edm_2d)

        # compute 3D EDM
        # edm_3d_gt = self.compute_edm(keypoints_3d) if not self.complete_edm else generate_edm(q=joint_angles, fixed_distances=False, robot=self.robot, graph=self.graph)
        # edm_3d_gt_upper = self.compute_edm_upper_triu(edm_3d_gt)

        # when loading precomputed seg masks
        mask=0
        if os.path.exists(mask_path):
            to_tensor = transforms.ToTensor()
            mask = self.pil_loader(mask_path, mode='L')
            mask = to_tensor(mask)
        # -------------------------------------#

        input_img = self.pil_loader(img_path)
        img = self.transform(input_img)    
        img_deeplab = self.deeplab_transforms(input_img)

        if "kinect360" in self.dataset:
            pose_gt = DREAM_Real_Poses.kinect360
        elif "realsense" in self.dataset:
            pose_gt = DREAM_Real_Poses.realsense
        elif "azure" in self.dataset:
            pose_gt = DREAM_Real_Poses.azure
        elif "orb" in self.dataset:
            pose_path = os.path.join(self.dataset_path, data_id + "_pose.npy")
            pose_gt = torch.from_numpy(np.load(pose_path)).squeeze().float()
            pose_gt = pose_gt[:3, :] # remove homogeneous row
        elif "panda" in self.dataset:
            pose_gt = DREAM_Real_Poses.kinect360
        
        fx_fy_render, cx_cy_render, img_size_render = compute_intrinsics_for_render_after_resize_and_crop(self.K, self.cfg)
        K_render = intrisic_matrix_from_parameters(fx_fy_render, cx_cy_render)

        # add noise to g.t. keypoints
        noise = np.random.normal(loc=0, scale=30, size=(7, 2)).astype(np.float32)
        keypoints_2d = keypoints_2d + noise

        sample = {
            'data_id': data_id,
            'img_path': img_path,
            'mask': mask,
            'img': img,
            'pose': pose_gt,
            'joint_angles_gt': torch.from_numpy(joint_angles),
            'edm_2d_upper': torch.from_numpy(edm_2d_upper),
            'keypoints_3d_gt': torch.from_numpy(keypoints_3d),
            'keypoints_2d_gt': torch.from_numpy(keypoints_2d),
            'robot_base_2d': torch.from_numpy(robot_base_2d),
            'K_orig': self.K,
            'K_render': K_render,
            'fx_fy_render': fx_fy_render,
            'cx_cy_render': cx_cy_render,
            'img_size_render': torch.Tensor(img_size_render),
            'img_deeplab': img_deeplab, # used for eval with deeplab seg model
            # 'img_orig': transforms.ToTensor()(input_img),
        }
            
        return sample

    def __len__(self):
        return len(self.frames)
    
    def get_cam_intrinsics_after_resize_and_crop(self, cfg):
        focal_length, principal_point, img_size = compute_intrinsics_for_render_after_resize_and_crop(self.K, cfg)
        return focal_length, principal_point, img_size
    
    def get_cam_intrinsics(self):
        return self.K
    
    def pil_loader(self, path, mode='RGB'):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with Image.open(path) as img:
            if mode == 'RGB':
                try:
                    return img.convert('RGB')
                except:
                    print(f"Error converting {path} to RGB")
                    assert 2 == 1
            elif mode == 'L':
                return img.convert('L')

    def compute_edm(self, keypoints):
        edm = np.sum((keypoints[:, np.newaxis, :] - keypoints[np.newaxis, :, :]) ** 2, axis = -1)
        assert np.array_equal(edm, edm.T), "edm and edm.T are not equal !"
        return edm

    def compute_edm_upper_triu(self, edm):
        edm_triu = np.triu(edm, k=1)
        x, y = np.triu_indices(edm.shape[0], k=1)
        edm_upper = edm_triu[x, y]
        # edm_upper = edm_triu[np.newaxis, edm_triu != 0.]
        return edm_upper

    def adjust_robot_definition(self, keypoints_2d, keypoints_3d):
        # ----- REMOVE PANDA HAND ------#
        keypoints_2d = keypoints_2d[1:]
        keypoints_3d = keypoints_3d[1:]

        # set joints 1 and 2 to the same origin
        keypoints_2d[0] = keypoints_2d[1]
        keypoints_3d[0] = keypoints_3d[1]

        # set joints 5 and 6 to the same origin
        keypoints_2d.insert(4, keypoints_2d[4])
        keypoints_3d.insert(4, keypoints_3d[4])
        return keypoints_2d, keypoints_3d

    def get_robot(self):
        return self.robot


class EDMDatasetSynthetic(Dataset):
    def __init__(self, root, dataset, complete_edm, robot_type, cfg):
        super(EDMDatasetSynthetic, self).__init__()
        self.cfg = cfg
        self.root = root
        self.dataset = dataset
        self.dataset_path = os.path.join(self.root, self.dataset)
        self.masks_dir = os.path.join(self.root, dataset+"_gt_masks")
        self.complete_edm = complete_edm
        self.robot_type = robot_type
        self.frames = sorted(os.listdir(self.dataset_path))
        self.frames = [frame for frame in self.frames if ".json" in frame]
        self.objects_file = self.frames[-1]
        self.camera_file = self.frames[-2]
        self.frames = self.frames[:-2]

        self.num_keypoints_per_image = 7
        self.num_distances_upper = (self.num_keypoints_per_image * (self.num_keypoints_per_image -1)) // 2

        self.is_panda = (self.robot_type == "panda")
        self.data_key = "panda_joint" if self.is_panda else "iiwa7_joint_" if self.is_kuka else float('nan') # only panda is supported for now

        self.robot, self.graph = load_panda_truncated(limits=None)

        self.panda_robot = PandaRobotArm(urdf_path=cfg.urdf_path, ee_name=cfg.ee_name)

        self.K = get_camera_intrinsics(os.path.join(self.dataset_path, self.camera_file))

        self.transform = transforms.Compose([
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
        ])

    def __getitem__(self, index):
        frame = self.frames[index]
        data_id = frame.split('.')[0]
        img_path = os.path.join(self.dataset_path, data_id + ".rgb.jpg")
        mask_path = os.path.join(self.masks_dir, data_id + "_gt_mask.jpg")

        with open(os.path.join(self.dataset_path, frame)) as f:
            data = json.load(f)
            keypoints = data['objects'][0]['keypoints']
            joint_data = data["sim_state"]["joints"]
            joint_angles = np.array([joint_data[i]['position'] for i in range(len(joint_data)) if self.data_key in joint_data[i]['name']], dtype=np.float32)[:7]
            keypoints_dict = [keypoints[i] for i in range(len(keypoints)) if self.data_key in keypoints[i]['name'] and len(keypoints[i]['name']) < 15]
            keypoints_sorted = sorted(keypoints_dict, key=lambda d: d['name'])
            keypoints_2d = np.array([keypoints_sorted[i]['projected_location'] for i in range(len(keypoints_sorted))], dtype=np.float32)
            keypoints_3d = np.array([keypoints_sorted[i]['location'] for i in range(len(keypoints_sorted))], dtype=np.float32)
            keypoints_3d /= 100 # cm to m conversion
            # joint_angles = joint_angles[:-1] # remove last joint that has value 0
            
            # remove panda hand
            keypoints_2d = keypoints_2d[:-1]
            keypoints_3d = keypoints_3d[:-1]

            input_img = self.pil_loader(img_path)
            img = self.transform(input_img)

            fx_fy_render, cx_cy_render, img_size_render = compute_intrinsics_for_render_after_resize_and_crop(self.K, self.cfg)
            K_render = intrisic_matrix_from_parameters(fx_fy_render, cx_cy_render)

            mask=0
            if os.path.exists(mask_path):
                to_tensor = transforms.ToTensor()
                mask = self.pil_loader(mask_path, mode='L')
                mask = to_tensor(mask)
            
            pose_path = os.path.join(self.dataset_path, data_id + "_pose.npy")
            pose_gt = torch.from_numpy(np.load(pose_path)).squeeze().float()
            pose_gt = pose_gt[:3, :] # remove homogeneous row

            robot_base_3d_obj = self.panda_robot.get_joint_3d_locations_from_angles(torch.from_numpy(joint_angles[np.newaxis, :])).squeeze(0)
            robot_base_3d_obj = robot_base_3d_obj[0]
            robot_base_3d_obj[2] = robot_base_3d_obj[2] - self.panda_robot.joint_01_length
            robot_base_2d = pose_gt[:3, :3] @ robot_base_3d_obj + pose_gt[:3, 3]
            robot_base_2d = robot_base_2d / robot_base_2d[2]
            robot_base_2d = (self.K @ robot_base_2d)[:2]

            # np.set_printoptions(suppress=True)
            if self.is_panda:
                assert np.allclose(keypoints_2d[0], keypoints_2d[1], rtol=1e-01, atol=1e-08) and np.allclose(keypoints_2d[4], keypoints_2d[5], rtol=1e-01, atol=1e-08), "For panda_synth data, joint1 and joint2, joint5 and joint6 keypoints should be equal, an example is found where it's NOT TRUE."
                # if not (np.allclose(keypoints_2d[0], keypoints_2d[1], rtol=1e-02, atol=1e-08) and np.allclose(keypoints_2d[4], keypoints_2d[5], rtol=1e-02, atol=1e-08)):# print(f"kp0: {keypoints_2d[0]}, kp1: {keypoints_2d[1]}")# print(f"kp4: {keypoints_2d[4]}, kp5: {keypoints_2d[5]}")

            edm_2d = self.compute_edm(keypoints_2d)
            edm_2d_upper = self.compute_edm_upper_triu(edm_2d)
            edm_3d_gt = self.compute_edm(keypoints_3d) if not self.complete_edm else self.generate_edm(q=joint_angles, fixed_distances=False, robot=self.robot, graph=self.graph)
            edm_3d_gt_upper = self.compute_edm_upper_triu(edm_3d_gt)

            sample = {
                'data_id': data_id,
                'img_path': img_path,
                'mask': mask,
                'img': img,
                'pose': pose_gt,
                'joint_angles_gt': torch.from_numpy(joint_angles[:6]),
                'edm_2d_upper': torch.from_numpy(edm_2d_upper),
                'keypoints_3d_gt': torch.from_numpy(keypoints_3d),
                'keypoints_2d_gt': torch.from_numpy(keypoints_2d),
                'robot_base_2d': robot_base_2d,
                'K_orig': self.K,
                'K_render': K_render,
                'fx_fy_render': fx_fy_render,
                'cx_cy_render': cx_cy_render,
                'img_size_render': torch.Tensor(img_size_render),
                # 'img_orig': transforms.ToTensor()(input_img)
            }
            
            return sample

    def __len__(self):
        return len(self.frames)

    def compute_edm(self, keypoints):
        edm = np.sum((keypoints[:, np.newaxis, :] - keypoints[np.newaxis, :, :]) ** 2, axis = -1)
        # edm = edm ** 0.5 if is_2d else edm
        assert np.array_equal(edm, edm.T), "edm and edm.T are not equal !"
        return edm

    def compute_edm_upper_triu(self, edm):
        edm_triu = np.triu(edm, k=1)
        x, y = np.triu_indices(edm.shape[0], k=1)
        edm_upper = edm_triu[x, y]
        # edm_upper = edm_triu[np.newaxis, edm_triu != 0.]
        return edm_upper
    
    def generate_edm(self, q=None, fixed_distances=True, robot=None, graph=None):

        if robot is None and graph is None:
            robot, graph = load_panda_truncated(limits=None)

        # NOTE labels of points (nodes in graph) arranged as in the adjacency matrix
        # node_ids = graph.node_ids

        q = list_to_variable_dict(q) if q is not None else robot.zero_configuration()

        # make corresponding distance matrix
        D = distance_matrix_from_graph(graph.realization(q))#[:-2,:-2]

        if fixed_distances:
            # adjacency matrix of graph of constant distances of robot
            A = graph.adjacency_matrix()#[:-2,:-2]
            # filter for known distances
            D_known = A*D
            return D_known, A
        else:
            return D
    
    
    def pil_loader(self, path, mode='RGB'):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with Image.open(path) as img:
            if mode == 'RGB':
                try:
                    return img.convert('RGB')
                except:
                    print(f"Error converting {path} to RGB")
                    assert 2 == 1
            elif mode == 'L':
                return img.convert('L')
            

class SegmentPandaDataset(Dataset):
    def __init__(self, root, dataset, cfg):
        super(SegmentPandaDataset, self).__init__()

        self.cfg = cfg
        self.root = root
        self.dataset = dataset
        self.dataset_path = os.path.join(self.root, self.dataset)
        self.masks_dir = os.path.join(self.root, dataset+"_gt_masks")
        self.frames = sorted(os.listdir(self.dataset_path))
        self.frames = [frame for frame in self.frames if ".json" in frame]
        self.objects_file = self.frames[-1]
        self.camera_file = self.frames[-2]
        self.frames = self.frames[:-2]

        self.transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
        self.to_tensor = transforms.ToTensor()
        
    
    def __getitem__(self, index):
        frame = self.frames[index]
        data_id = frame.split('.')[0]
        img_path = os.path.join(self.dataset_path, data_id + ".rgb.jpg")
        mask_path = os.path.join(self.masks_dir, data_id + "_gt_mask.jpg")

        input_img = self.pil_loader(img_path)
        img = self.transforms(input_img)

        mask = self.pil_loader(mask_path, mode='L')
        mask = self.to_tensor(mask)

        sample = {
            'data_id': data_id,
            # 'img_orig': transforms.ToTensor()(input_img),
            'img': img,
            'mask': mask,
        }
        return sample
        


    def __len__(self):
        return len(self.frames)
    
    def pil_loader(self, path, mode='RGB'):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with Image.open(path) as img:
            if mode == 'RGB':
                try:
                    return img.convert('RGB')
                except:
                    print(f"Error converting {path} to RGB")
                    assert 2 == 1
            elif mode == 'L':
                return img.convert('L')
            
    

class UR5SynthDataset(Dataset):
    def __init__(self, root, dataset, cfg):
        super(UR5SynthDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        self.dataset_path = os.path.join(self.root, self.dataset)
        self.n_dof = 6
        self.cfg = cfg
        self.img_size_render = [cfg.img_h_model_resize_and_crop, cfg.img_w_model_resize_and_crop]
        
        files = os.listdir(self.dataset_path)
        images = [f for f in files if f.endswith(".jpg")]
        self.num_samples = len(images)
        del images

        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data_id = f"{index:06}"
        img_path = os.path.join(self.dataset_path, data_id+".jpg")
        input_img = self.pil_loader(img_path, mode="L")
        img = self.transform(input_img)
        joint_angles = np.loadtxt(os.path.join(self.dataset_path, data_id+"_q.txt"))
        pts_3d_cam = np.loadtxt(os.path.join(self.dataset_path, data_id+"_pts_3d_cam.txt"))
        pose = np.loadtxt(os.path.join(self.dataset_path, data_id+"_pose.txt")) # row-wise

        sample = {
            'data_id': data_id,
            'img_path': img_path,
            'mask': img,
            'img': 0,
            'pose': torch.from_numpy(pose).float(),
            'joint_angles_gt': torch.from_numpy(joint_angles).float(),
            'edm_2d_upper': 0,
            'keypoints_3d_gt': torch.from_numpy(pts_3d_cam).float(),
            'keypoints_2d_gt': 0,
            'robot_base_2d': 0,
            'K_orig': 0,
            'K_render': 0,
            'fx_fy_render': 0,
            'cx_cy_render': 0,
            'img_size_render': torch.Tensor(self.img_size_render),
        }

        return sample



    def __len__(self):
        return self.num_samples
    

    def pil_loader(self, path, mode='RGB'):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with Image.open(path) as img:
            if mode == 'RGB':
                try:
                    return img.convert('RGB')
                except:
                    print(f"Error converting {path} to RGB")
                    assert 2 == 1
            elif mode == 'L':
                return img.convert('L')