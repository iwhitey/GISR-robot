import torch
import torchvision

from PIL import Image


def crop_and_resize(img, keypoints_2d, output_size=(240, 320)):

    batch_size = img.shape[0]
    img_h = img.shape[-2]
    img_w = img.shape[-1]

    min_x = torch.clamp(keypoints_2d[:,:,0].min(axis=1)[0] - 0.2 * img_w, min=0)
    max_x = torch.clamp(keypoints_2d[:,:,0].max(axis=1)[0] + 0.2 * img_w, max=img_w)
    min_y = torch.clamp(keypoints_2d[:,:,1].min(axis=1)[0] - 0.2 * img_h, min=0)
    max_y = torch.clamp(keypoints_2d[:,:,1].max(axis=1)[0] + 0.2 * img_h, max=img_h)
    
    top_left = torch.cat([min_x.unsqueeze(1), min_y.unsqueeze(1)], axis=1)
    bottom_right = torch.cat([max_x.unsqueeze(1), max_y.unsqueeze(1)], axis=1)
    centroid = keypoints_2d.mean(axis=1)

    d_x = torch.maximum((top_left[:,0] - centroid[:,0]).abs(), (bottom_right[:,0] - centroid[:,0]).abs())
    d_y = torch.maximum((top_left[:,1] - centroid[:,1]).abs(), (bottom_right[:,1] - centroid[:,1]).abs())

    lamda = 1.4; r = 4/3
    crop_w = torch.maximum(d_x, d_y / r) * 2 * lamda
    crop_h = torch.maximum(d_x / r, d_y) * 2 * lamda

    min_x = centroid[:,0] - crop_w/2; min_y = centroid[:,1] - crop_h/2; max_x = centroid[:,0] + crop_w/2; max_y = centroid[:,1] + crop_h/2
    bounding_boxes = torch.cat([min_x.unsqueeze(1), min_y.unsqueeze(1), max_x.unsqueeze(1), max_y.unsqueeze(1)], axis=1)
    bounding_boxes = torch.cat([torch.arange(img.shape[0], device='cuda').unsqueeze(1), bounding_boxes], axis=1)
    crops = torchvision.ops.roi_align(img, bounding_boxes, output_size=output_size, sampling_ratio=4)

    return crops, bounding_boxes[:, 1:]


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    with Image.open(path) as img:
        return img.convert('RGB')