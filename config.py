from dataclasses import dataclass, field
from typing import Union, List
import os


@dataclass
class GIMConfig:
    use_gim: bool = False           # whether to use the geometric initialization module
    rotation_from_pnp: bool = False
    scale_from_kinematic_params: bool = False # if True, the translation scale is initialized from known robot link lengths and 2D keypoints
    scale_from_pnp: bool = False  # if True, the translation scale is initialized from pnp (also requires 2D keypoints)
    init_robot_configuration: bool = False # if True, gim provides initial robot configuration using the edm_model
    edm_model_path: str = "checkpoints/panda_edm_model.pth"  # path to the EDM model checkpoint, required if init_robot_configuration is True


@dataclass
class TrainConfig:
    train_data_root: str = "."            # e.g. "path_to_dream_dataset_synthetic_root_dir"
    train_dataset: Union[str, list] = "panda-3cam_azure_no_occlusions_reduced" # a single dataset folder under data root e.g. "panda_synth_train_dr", or a list of folders e.g. ["panda_synth_1", "panda_synth_2"] 
    val_data_root: str = "."              # same logic as train data
    val_dataset: Union[str, list] = "panda-3cam_azure_no_occlusions_reduced"
    n_iter: int = 3                         # max. number of refinement iterations during training
    gim_config: GIMConfig = field(default_factory=GIMConfig) # config for geometric init. module
    device: str = "cuda"
    lr: float = 3e-4
    batch_size: int = 32
    grad_accumulation_steps: int = 1    # if >1, gradient accumulation is used
    num_epochs: int = 700
    warmup_epochs: int = 30
    clip_grad_norm: float = 10.0
    save_model_freq: int = 50
    validation_freq: int = 1
    load_model_checkpoint: bool = False # whether to load a model checkpoint to resume training
    load_ddp_model: bool = False    # whether the loaded model was trained with DDP
    checkpoint_path: str = ""       # path to model checkpoint if load_model_checkpoint is True
    save_model_path: str = "checkpoints/test"       # path to save model checkpoints and train log
    img_width_orig: int = 640        # input image original sizes
    img_height_orig: int = 480
    img_h_model_resize: int = 236    # sizes after resizing
    img_w_model_resize: int = 236
    img_h_model_resize_and_crop: int = 224  # sizes after resizing and cropping (this is the input size to the model)
    img_w_model_resize_and_crop: int = 224
    robot: str  = ""        # these fields are robot specific and populated based on provided robot type
    urdf_path: str = ""
    ee_name: str = ""
    mesh_base_dir: str = ""
    mesh_files: list = field(default_factory=list)

@dataclass
class EvalConfig:
    data_root: str = "" # e.g. "path_to_dream_dataset_root_dir or ur5_root_dir"
    dataset: Union[str, List] = ""  # a single dataset folder under data root e.g. "panda_synth_train_dr", or a list e.g. ["panda_synth_1", "panda_synth_2"]
    n_iter: int = 3
    gim_config: GIMConfig = field(default_factory=GIMConfig) # config for geometric init. module
    batch_size: int = 128
    device: str = "cuda"
    checkpoint_path: str = "checkpoints/panda_refiner.pth"    # path to refiner model checkpoint
    load_ddp_model: bool = True    # whether the loaded refiner model was trained with DDP
    use_segmentation_model: bool = False   # if seg model is not used, then masks are rendered from g.t. robot data or loaded as precomputed masks as in dataset.py
    seg_checkpoint_path: str = "checkpoints/panda_segmentation.pth"
    log_path: str = "logs/eval_log.txt"
    img_width_orig: int = 640       # input image original sizes
    img_height_orig: int = 480
    img_h_model_resize: int = 236   # sizes after resizing
    img_w_model_resize: int = 236
    img_h_model_resize_and_crop: int = 224 # sizes after resizing and cropping (this is the input size to the model)
    img_w_model_resize_and_crop: int = 224
    robot: str = ""       # these fields are robot specific and populated based on provided robot type
    urdf_path: str = ""
    ee_name: str = ""
    mesh_base_dir: str = ""
    mesh_files: list = field(default_factory=list)


def get_robot_info(robot_name):
    """
    If the URDF files were relocated after cloning the repo, update these paths accordingly.
    """
    if robot_name == "panda":
        urdf_path = "urdfs/Panda/panda.urdf"
        ee_name = "panda_hand"
        mesh_base_dir = "urdfs/Panda/meshes/visual/"
        mesh_files = [mesh_base_dir + "link0/link0.obj",
                    mesh_base_dir + "link1/link1.obj",
                    mesh_base_dir + "link2/link2.obj",
                    mesh_base_dir + "link3/link3.obj",
                    mesh_base_dir + "link4/link4.obj",
                    mesh_base_dir + "link5/link5.obj",
                    mesh_base_dir + "link6/link6.obj",
                    mesh_base_dir + "link7/link7.obj",
                    mesh_base_dir + "hand/hand.obj",
                    ]
        
    elif robot_name == "ur5":
        urdf_path = "urdfs/ur_description/urdf/universalUR5.urdf"
        ee_name = "wrist_3_link"
        mesh_base_dir = "urdfs/ur_description/meshes/ur5/visual"
        mesh_names = ['base.obj', 'shoulder.obj', 'upperarm.obj', 'forearm.obj', 'wrist1.obj', 'wrist2.obj', 'wrist3.obj']
        mesh_files = [os.path.join(mesh_base_dir, mesh_file_name) for mesh_file_name in mesh_names]

    return urdf_path, ee_name, mesh_base_dir, mesh_files
    
def get_train_cfg(args):

    urdf_path, ee_name, mesh_base_dir, mesh_files = get_robot_info(args.robot)

    cfg = TrainConfig(
        robot=args.robot,
        urdf_path=urdf_path,
        ee_name=ee_name,
        mesh_base_dir=mesh_base_dir,
        mesh_files=mesh_files
    )

    check_consistent_train_cfg(cfg)

    print("logging the training process to ", cfg.save_model_path)
    cfg.log_path = os.path.join(cfg.save_model_path, "train_log.txt")
    os.makedirs(cfg.save_model_path, exist_ok=True)
    return cfg


def get_eval_cfg(args):

    urdf_path, ee_name, mesh_base_dir, mesh_files = get_robot_info(args.robot)

    cfg = EvalConfig(
        robot=args.robot,
        urdf_path=urdf_path,
        ee_name=ee_name,
        mesh_base_dir=mesh_base_dir,
        mesh_files=mesh_files,
     )

    check_consistent_eval_cfg(cfg)
    eval_log_dir = os.path.dirname(cfg.log_path)
    if eval_log_dir:
        os.makedirs(eval_log_dir, exist_ok=True)
    return cfg



def check_consistent_train_cfg(cfg):
    if cfg.robot not in ["panda", "ur5"]:
        raise ValueError("Unsupported robot. Supported robots are: panda, ur5")
    if cfg.gim_config.use_gim and cfg.robot != "panda":
        raise ValueError("GIM is currently only supported for panda robot")
    if cfg.gim_config.use_gim and cfg.gim_config.init_robot_configuration and cfg.gim_config.edm_model_path == "":
        raise ValueError("If GIM is set to estimate robot configuration, edm_model_path must be provided")
    if cfg.gim_config.use_gim and cfg.gim_config.scale_from_kinematic_params and cfg.gim_config.scale_from_pnp:
        raise ValueError("scale initialization can be done either from kinematic params or pnp, not both")
    if cfg.load_model_checkpoint and cfg.load_model_path == "":
        raise ValueError("If load_model_checkpoint is True, load_model_path must be provided")
    if cfg.save_model_path == "":
        raise ValueError("save_model_path must be provided for model checkpointing")
    if not cfg.gim_config.init_robot_configuration:
        cfg.gim_config.edm_model_path = ""

def check_consistent_eval_cfg(cfg):
    if cfg.robot not in ["panda", "ur5"]:
        raise ValueError("Unsupported robot. Supported robots are: panda, ur5")
    if cfg.gim_config.use_gim and cfg.robot != "panda":
        raise ValueError("GIM is currently only supported for panda robot")
    if cfg.gim_config.use_gim and cfg.gim_config.init_robot_configuration and cfg.gim_config.edm_model_path == "":
        raise ValueError("If GIM is set to estimate robot configuration, edm_model_path must be provided")
    if cfg.use_segmentation_model and cfg.seg_checkpoint_path == "":
        raise ValueError("If use_segmentation_model is True, seg_checkpoint_path must be provided")
    if cfg.gim_config.use_gim and cfg.gim_config.scale_from_kinematic_params and cfg.gim_config.scale_from_pnp:
        raise ValueError("scale initialization can be done either from kinematic params or pnp, not both")
    if not cfg.gim_config.init_robot_configuration:
        cfg.gim_config.edm_model_path = ""
