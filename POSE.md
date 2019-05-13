# Pose estimation

## Notes

**Top-down**: Incorporate person detector followed by pose estimation.
**Bottom-up**: Detect all parts in the image followed by part association.
**Discriminative**: Directly predict hand pose from image features.
**Generative**: Minimize discrepancy between hand model and hand pose.


## Papers

1. [RMPE: Regional Multi-Person Pose  Estimation](#rmpe)

<details>
<summary>
RMPE: Regional Multi-Person Pose Estimation
</summary>
<p>
[Fang, 2018] [paper](https://arxiv.org/pdf/1612.00137.pdf)

In this paper, the authors argue that accuracy of person detectors is crcucial to the success of top-down pose estimation methods and that slight localization errors can cause large failures. 

The authors propose using a Symmetric Spatial Transformer Network (SSTN) to extract high quality single person regions from inaccurate bounding boxes. A Single Person Pose Estimator (SPPE) is then used to estimate the human pose together with a Paralle (SPPE) that  
</p>
</details>





