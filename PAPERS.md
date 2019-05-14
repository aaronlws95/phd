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

<details>
<summary>
Disentangling Latent Hands for Image Synthesis and Pose Estimation
</summary>
<p>
<a href="https://arxiv.org/pdf/1812.01002.pdf">paper</a>

In this paper, the authors propose using a disentangled variational autoencoder (dVAE) to deal with the problem of large variation of factors associated with RGB images for hand pose estimation or hand image synthesis. The dVAE is used to learn disentangled representations from these images. 

For the hand pose estimation example, the authors aim to predict the 3D pose (3DPose), canonical pose (CPose) and viewpoint from an RGB image. They embded the 3DPose and the RGB image into a shared latent space and learn the disentangled representation using the dVAE. Inference is carried out by encoding the RGB image as a latent variable and disentangling into the CPose, 3DPose, and viewpoint. With this method, they are able to leverage unlabelled or weakly labelled data. 
</p>
</details> 

<details>
<summary>
Point-to-Pose Voting based Hand Pose Estimation using Residual Permutation
Equivariant Layer
</summary>
<p>
<a href="https://arxiv.org/pdf/1812.02050.pdf">paper</a>

In this paper, the authors propose a method for hand pose estimation from an unordered point cloud. The method uses a residual Permutation Equivariant Layer (PEL) and a voting-based scheme.

First, the 3D points are view normalized such that the view direction is pointed towards the hand centroid. This provides us with a one-to-one input-output mapping. The residual PEL then computes separate features for each individual point. Finally, with the local point-wise features, the hand pose is estimated using a point-to-pose voting scheme. Two versions are presented: detection and regression.

For the detection point-to-pose voting scheme, two separate fully connected modules are used to estimate an importance matrix and a distribution matrix. The importance matrix consists of confidence levels for the n-th input to predict the j-th output pose dimension. The distribution matrix is the probability distribution of each pose dimension. In other words, each of the N points predicts J (number of keypoints) B-dimensional distribtuions and J corresponding importance weights. The final pose is obtained by first merging the predictions of all N points into a final distribution folowed by integration over the distribution. (For regression, distribution estimation is not necessary). The authors also show that the importance term can also be used to carry out hand segmentation.
</p>
</details> 

<details>
<summary>
H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions
</summary>
<p>
<a href="https://arxiv.org/pdf/1904.05349.pdf">paper</a>

In this paper, the authors present a unified framework for joint estimation of 3D hand pose and 6D object pose and also recognition of object and action classes. The architecture only requires a single feed-forward pass and does not rely on external detection algorithms. 

In this method, the RGB image is passed through a convolutional network to produce a 3D grid where each cell contains a prediction of the hand pose, object pose, corresponding confidence levels, object class probabilities and action class probabilities. The predictions from the cell with the highest confidence for hand and object poses are chosen for each frame and then propagated in the temporal domain using LSTMs to account for long-term dependencies for action recognition. The dependencies between hand and object poses are modelled by a multilayer perceptron before being passed as input to the LSTM.
</p>
</details> 