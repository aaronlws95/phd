# Pose estimation

## Notes

**Top-down**: Incorporate person detector followed by pose estimation.
**Bottom-up**: Detect all parts in the image followed by part association.
**Discriminative**: Directly predict hand pose from image features.
**Generative**: Minimize discrepancy between hand model and hand pose.


## Papers

<!-- Template -->
<!-- 
<details>
<summary>
</summary>
<p>
</p>
</details> 
-->

<details>
<summary>
RMPE: Regional Multi-Person Pose Estimation
</summary>
<p>
<a href="https://arxiv.org/pdf/1612.00137.pdf">paper</a>

In this paper, the authors argue that accuracy of person detectors is crcucial to the success of top-down pose estimation methods and that slight localization errors can cause large failures.

The authors propose using a Symmetric Spatial Transformer Network (SSTN) to extract high quality single person regions from inaccurate bounding boxes. A Single Person Pose Estimator (SPPE) is then used to estimate the human pose together with a Parallel SPPE that compares the centered labels. A Spatial De-transformer Network (SDTN) is included to remap the estimated human pose back to the original image coordinate. 

To reduce redundant pose estimations, the authors also included a parametric pose non-maximum suppression (NMS). Additionally, they propose a Pose-guided Proposals Generator to augment the training data by learning the conditional distribution of bounding boxes for a given human pose.

The authors present their method on 2D multiple human pose estimation on the MPII dataset.
</p>
</details>

<details>
<summary>
OpenPose: Realtime Multi-Person 2D Pose 
Estimation using Part Affinity Fields
</summary>
<p>
<a href="https://arxiv.org/pdf/1812.08008.pdf">paper</a>

In this paper, the authors present a bottom-up approach to 2D multiple human pose estimation. They claim that top-down approaches relies too much on the person detector and that performance is proportional to number of people in the image. 

The approach begins by generating feature maps using a standard convolutional network. After that the feature maps are used to estimate Part Affinity Fields (PAF) for part association and Part Confidence Maps for part detection. The network architecture is iterative and refines the predictions over multiple stages.

Bipartite graphs are formed between the parts obtained from the confidence maps where the PAFs are used to prune lower scored connections.
</p>
</details>

<details>
<summary>
On Pre-Trained Image Features and Synthetic Images
for Deep Learning
</summary>
<p>
<a href="http://openaccess.thecvf.com/content_ECCVW_2018/papers/11129/Hinterstoisser_On_Pre-Trained_Image_Features_and_Synthetic_Images_for_Deep_Learning_ECCVW_2018_paper.pdf">paper</a>

In this paper, the authors present a method to improve training of object detection using only synthetic data based on a network pre-trained on real data. The authors suggest freezing the pre-learned weights trained on real images responsible for feature extraction and train the remaining weights only on synthetic images. 
</p>
</details> 
