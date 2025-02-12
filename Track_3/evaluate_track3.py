import argparse
import glob
import numpy as np
import os
import torch
import torch.nn as nn
from functools import wraps
from torch import Tensor
from typing import Any, Callable, Optional, Sequence, Union
import warnings
from pathlib import Path
from PIL import Image
from torch.utils.cpp_extension import load
# import debugpy
# debugpy.listen(5679)
# debugpy.wait_for_client()
try:
    root = Path(__file__).parent
    cd = load(
        name='cd',
        sources=[
            root/'chamfer_distance.cpp',
            root/'chamfer_distance.cu',
        ]
    )
except (RuntimeError, IndexError) as e:  # Typically due to Ninja/CUDA being unavailable
    warnings.warn(f'Chamfer Distance module unavailable: {e}')
    cd = None

K = np.array([[1591.1095,0.0,945.64465,0],[0.0, 1591.1095,722.8234,0],[0,0,1,0],[0,0,0,1]])

def compute_errors(gt, pred, conf = None):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt_og = gt.copy()
    pred_og = pred.copy()
    if conf is not None:
        assert (conf>=1).sum() > 0
        gt =  gt[conf>=1]
        pred = pred[conf>=1]
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2

    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    
    if conf is not None:
        if np.array(conf == 2).sum() == 0:
            mask = np.array(conf == 1)
        else:
            mask = np.array(conf == 2)
    else:
        mask = np.ones_like(gt_og).astype(bool)
    mp = metrics_pointcloud(pred_og, gt_og, mask, K)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, mp['F-Score']
class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if cd is None:
            raise RuntimeError(f'Chamfer Distance module unavailable')

    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)
    
def get_device(device: Optional[Union[str, torch.device]]):
    """Create torch device from str or device. Defaults to CUDA if available."""
    if isinstance(device, torch.device): return device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def map_container(f: Callable) -> Callable:
    """Decorator to recursively apply a function to arbitrary nestings of `dict`, `list`, `tuple` & `set`

    NOTE: `f` can have an arbitrary signature, but the first arg must be the item we want to apply `f` to.

    Example:
    ```
        @map_apply
        def square(n, bias=0):
            return (n ** 2) + bias

        x = {'a': [1, 2, 3], 'b': 4, 'c': {1: 5, 2: 6}}
        print(map_apply(x))

        ===>
        {'a': [1, 4, 9], 'b': 16, 'c': {1: 25, 2: 36}}

        print(map_apply(x, bias=2))

        ===>
        {'a': [3, 6, 11], 'b': 18, 'c': {1: 27, 2: 38}}
    ```
    """
    @wraps(f)
    def wrapper(x: Any, *args, **kwargs) -> Any:
        if isinstance(x, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in x.items()}

        elif isinstance(x, list):
            return [wrapper(v, *args, **kwargs) for v in x]

        elif isinstance(x, tuple):
            return tuple(wrapper(v, *args, **kwargs) for v in x)

        elif isinstance(x, set):
            return {wrapper(v, *args, **kwargs) for v in x}

        else:  # Base case, single item
            return f(x, *args, **kwargs)

    return wrapper

@map_container
def to_torch(x: Any, permute: bool = True, device: Optional[torch.device] = None):
    """Convert given input to torch.Tensors

    :param x: (Any) Arbitrary structure to convert to tensors (see `map_apply`).
    :param permute: (bool) If `True`, permute to PyTorch convention (b, h, w, c) -> (b, c, h, w).
    :param device: (torch.device) Device to send tensors to.
    :return: (Any) Input structure, converted to tensors.
    """
    # Classes that should be ignored
    # if isinstance(x, (str, Timer, MultiLevelTimer)): return x

    # NOTE: `as_tensor` allows for free numpy conversions
    x = torch.as_tensor(x, device=device)

    if permute and x.ndim > 2:
        dim = [-1, -3, -2]  # Transpose last 3 dims as (2, 0, 1)
        dim = list(range(x.ndim - 3)) + dim  # Keep higher dimensions the same
        x = x.permute(dim)

    return x

def to_float(fn):
    """Helper to convert all metrics into floats."""
    @wraps(fn)
    def wrapper(*a, **kw):
        return {k: float(v) for k, v in fn(*a, **kw).items()}
    return wrapper
# POINTCLOUD
# -----------------------------------------------------------------------------
def _metrics_pointcloud(pred: Tensor, target: Tensor, th: float):
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).float().mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).float().mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3): return P, P  # No points are correct.

    f = 2*P*R / (P + R)
    iou = P*R / (P + R - (P*R))
    return f, iou


@to_float
def metrics_pointcloud(pred, target, mask, K):
    """Compute pointcloud-based prediction metrics.
    From Ornek: (https://arxiv.org/abs/2203.08122)

    These metrics are computed on the GPU, since Chamfer distance has quadratic complexity.
    Following the original _paper, we set the default threshold of a correct point to 10cm.
    An extra threshold is added at 20cm for informative purposes, but is not typically reported.

    :param pred: (ndarray) (h, w) Predicted depth.
    :param target: (ndarray) (h, w) Ground truth depth.
    :param mask: (ndarray) (h, w) Mask of valid pixels.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    device = get_device('cuda')
    pred, target, K = to_torch((pred, target, K.astype(np.float32)), device=device)
    K_inv = K.inverse()[None]

    backproj = BackprojectDepth(pred.shape).to(device)
    pred_pts = backproj(pred[None, None], K_inv)[:, :3, mask.flatten()]
    target_pts = backproj(target[None, None], K_inv)[:, :3, mask.flatten()]

    pred_nn, target_nn = ChamferDistance()(pred_pts.permute(0, 2, 1), target_pts.permute(0, 2, 1))
    pred_nn, target_nn = pred_nn.sqrt(), target_nn.sqrt()

    f1, iou1 = _metrics_pointcloud(pred_nn, target_nn, th=0.1)
    f2, iou2 = _metrics_pointcloud(pred_nn, target_nn, th=0.2)
    return {
        'Chamfer': pred_nn.mean() + target_nn.mean(),
        'F-Score': 100 * f1,
        'IoU': 100 * iou1,
        'F-Score-20': 100 * f2,
        'IoU-20': 100 * iou2,
    }
# -----------------------------------------------------------------------------
class BackprojectDepth(nn.Module):
    """Module to backproject a depth map into a pointcloud.

    :param shape: (tuple[int, int]) Depth map shape as (height, width).
    """
    def __init__(self, shape):
        super().__init__()
        self.h, self.w = shape
        self.ones = nn.Parameter(torch.ones(1, 1, self.h*self.w), requires_grad=False)

        grid_w, grid_h = torch.meshgrid(torch.arange(self.w), torch.arange(self.h))  # (h, w), (h, w)
        grid = (grid_w, grid_h)
        pix = torch.stack(grid).view(2, -1)[None]  # (1, 2, h*w) as (x, y)
        pix = torch.cat((pix, self.ones), dim=1)  # (1, 3, h*w)
        self.pix = nn.Parameter(pix, requires_grad=False)

    def forward(self, depth: Tensor, K_inv: Tensor) -> Tensor:
        """Backproject a depth map into a pointcloud.

        Camera is assumed to be at the origin.

        :param depth: (Tensor) (b, 1, h, w) Depth map to backproject.
        :param K_inv: (Tensor) (b, 4, 4) Inverse camera intrinsic parameters.
        :return: (Tensor) (b, 4, h*w) Backprojected 3-D points as (x, y, z, homo).
        """
        b = depth.shape[0]
        pts = K_inv[:, :3, :3] @ self.pix.repeat(b, 1, 1)  # (b, 3, h*w) Cam rays.
        pts *= depth.flatten(-2)  # 3D points.
        pts = torch.cat((pts, self.ones.repeat(b, 1, 1)), dim=1)  # (b, 4, h*w) Add homogenous.
        return pts
    
def evaluate_track3(result, depth_path, gt_path):
    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Make sure the directory exists
    os.makedirs(depth_path, exist_ok=True)

    # Save each tuple as a .npy file in the specified directory
    for i, sub_array in enumerate(result):
        # Generate the file name with 4-digit zero padding
        file_name = f"{(i):04d}.npy"

        # Full path for the file
        file_path = os.path.join(depth_path, file_name)

        # Save the tuple to the specified path
        np.save(file_path, sub_array)

        print(f"Saved: {file_path}")


    # Get the list of image files to process
    if os.path.isfile(depth_path):
        if depth_path.endswith('txt'):
            with open(depth_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [depth_path]
    else:
        filenames = sorted(glob.glob(os.path.join(depth_path, '**/*'), recursive=True))

    if os.path.isfile(gt_path):
        if gt_path.endswith('txt'):
            with open(gt_path, 'r') as f:
                gt_filenames = f.read().splitlines()
        else:
            gt_filenames = [gt_path]
    else:
        gt_filenames = sorted(glob.glob(os.path.join(gt_path, '**/*'), recursive=True))

    # Process each image file
    errors = []
    for k, file in enumerate(zip(filenames, gt_filenames)):
        filename = file[0]
        gt_filename = file[1]
        print(f'Processing {k+1}/{len(filenames)}: Pred_depth: {filename}, GT_depth: {gt_filename}')
        
        # Load the predicted depth
        pred = np.load(filename)
        assert pred.ndim == 4, "pred depth shape should be (Batch, 1, Height, Width)"
        pred = pred[0, 0]
        height, width = pred.shape
        
        # Load the gt depth
        gt_depth_raw = np.array(Image.open(gt_filename))
        gt_depth_conf = np.array(Image.open(os.path.join(gt_filename.split('raw')[0]+'conf', gt_filename.split('/')[-1])))
        # mm to meter
        gt_depth_raw_m = gt_depth_raw.astype(np.float32) / 1000.0        
       
        # Resize the gt (192,256) -> (480,640)
        if gt_depth_raw_m.shape != pred.shape:
            gt_eval = Image.fromarray(gt_depth_raw_m)
            gt_eval_resize = gt_eval.resize((width, height), Image.NEAREST)
            gt_eval_resize = np.array(gt_eval_resize)
            gt_depth_conf_eval = Image.fromarray(gt_depth_conf)
            gt_depth_conf_eval_resize = gt_depth_conf_eval.resize((width, height), Image.NEAREST)
            gt_depth_conf_eval_resize = np.array(gt_depth_conf_eval_resize).astype(np.uint8)
        else:
            gt_eval_resize = gt_depth_raw_m
            gt_depth_conf_eval_resize = gt_depth_conf

        gt_eval_resize[gt_eval_resize == 0] = 1e-5
        pred[pred == 0] = 1e-5

        # Rescaled based on the median value
        ratio = np.median(gt_eval_resize) / np.median(pred)
        pred_eval = ratio * pred
        
        errors.append(compute_errors(gt_eval_resize, pred_eval, conf=gt_depth_conf_eval_resize))
        

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "fscore"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    return mean_errors.tolist()[-1]
