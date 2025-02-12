# LPCVC 2025 Evaluation Functions
This repository contains evaluation functions for the three tracks in the LPCVC 2025 Challenge. Each track has a specific evaluation metric.

## Track 1: Classification Accuracy
- Task: Image classification for different lighting conditions and styles.
- Metric: Accuracy Percentage
- Formula: 

$$\text{Accuracy} = \left( \frac{\text{Number of Correct Predictions}}{\text{Total Number of Samples}} \right)$$

## Track 2: Segmentation Quality (mIoU)
- Task: Open-vocabulary segmentation with text prompt.
- Metric: Mean Intersection over Union (mIoU)
- Formula:

$$\text{IoU} = \frac{\text{Intersection of Prediction and Ground Truth}}{\text{Union of Prediction and Ground Truth}}$$

$$\text{mIoU} = \frac{1}{N} \sum_{i=1}^{N} \text{IoU}_i$$

## Track 3: Depth Estimation (F-Score for Point Clouds)
- Task: Monocular relative depth estimation.
- Metric: Average F-Score based on point cloud precision and recall
- Formula: Given two sets of points,  (predicted depth points) and  (ground truth depth points), the precision  and recall  are computed based on a threshold distance . The F-score is calculated as:

$$F = \frac{2 \cdot P \cdot R}{P + R}$$

