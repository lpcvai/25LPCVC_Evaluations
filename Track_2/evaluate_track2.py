import os
import numpy as np
from glob import glob

def evaluate_track2(output_array, ground_truth_dir):
    # Load ground truth masks
    ground_truth_files = sorted(glob(os.path.join(ground_truth_dir, '*.npy')))
    
    if len(output_array) != len(ground_truth_files):
        raise ValueError("Number of predictions does not match the number of ground truth files.")

    mIoU_list = []

    for pred_mask, gt_file in zip(output_array, ground_truth_files):
        # Load ground truth mask
        gt_mask = np.load(gt_file)

        # Ensure prediction is also a binary mask
        print(pred_mask.shape)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Compute intersection and union
        I = (pred_mask & gt_mask).sum()
        U = (pred_mask | gt_mask).sum()

        # Compute IoU for this sample
        IoU = I / (U + 1e-6)

        # Convert to percentage
        mIoU_list.append(IoU.item())

    # Compute the mean IoU over all samples
    mean_mIoU = sum(mIoU_list) / len(mIoU_list)

    return mean_mIoU, mIoU_list  # Return mean IoU and list of per-sample IoUs