# Hand Pose Estimation

## [Wan 2019](http://www.vision.ee.ethz.ch/~wanc/papers/cvpr2019.pdf)

**Self-supervised 3D hand pose estimation from depth maps using a spherical hand model and multi-view supervision**.
```
@inproceedings{wan2019self,
  title={Self-supervised 3D hand pose estimation through training by fitting},
  author={Wan, Chengde and Probst, Thomas and Gool, Luc Van and Yao, Angela},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={10853--10862},
  year={2019}
}
```

## [Rad 2018](https://arxiv.org/pdf/1712.03904.pdf)

**3D hand pose estimation from depth by feature mapping real image features to synthetic features and joint training on real and synthetic images**.
```
@inproceedings{rad2018feature,
  title={Feature mapping for learning fast and accurate 3d pose inference from synthetic images},
  author={Rad, Mahdi and Oberweger, Markus and Lepetit, Vincent},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4663--4672},
  year={2018}
}
```

## [Boukhayma 2019](https://arxiv.org/pdf/1902.03451.pdf)

**Joint training on 3D and 2D hand annotations with reprojection to predict 3D hand shape (mesh) and pose from RGB**.
```
@inproceedings{boukhayma20193d,
  title={3d hand shape and pose from images in the wild},
  author={Boukhayma, Adnane and Bem, Rodrigo de and Torr, Philip HS},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={10843--10852},
  year={2019}
}
```

## [Cai 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yujun_Cai_Weakly-supervised_3D_Hand_ECCV_2018_paper.pdf)

**Weak supervision of 3D hand pose estimation from RGB by generating depth maps from predicted 3D poses**.
```
@inproceedings{cai2018weakly,
  title={Weakly-supervised 3d hand pose estimation from monocular rgb images},
  author={Cai, Yujun and Ge, Liuhao and Cai, Jianfei and Yuan, Junsong},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={666--682},
  year={2018}
}
```

## [Mueller 2018](https://arxiv.org/pdf/1712.01057.pdf)

**Translate synthetic images to real using GAN for 3D hand pose estimation from RGB**.

```
@inproceedings{mueller2018ganerated,
  title={Ganerated hands for real-time 3d hand tracking from monocular rgb},
  author={Mueller, Franziska and Bernard, Florian and Sotnychenko, Oleksandr and Mehta, Dushyant and Sridhar, Srinath and Casas, Dan and Theobalt, Christian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={49--59},
  year={2018}
}
```

## [Hasson 2019](https://arxiv.org/pdf/1904.05767.pdf)

**Joint reconstruction of object and hand mesh from monocular RGB**. Method employs two main branches, one for hand and one for object. For the hand branch, the MANO model (PCA) which maps a pose and shape parameter to a mesh is integrated as a differentiable layer. For the object branch, Atlasnet is employed for object prediction. Scale and translation is also predicted to keep object position and scale relative to the hand. Repulsion loss is introduced  to penalize interpentration and an attraction loss is introduced to penalize cases where the surfaces are not in contact. Creates own synthetic dataset for training and evaluation.

```
@inproceedings{hasson2019learning,
  title={Learning joint reconstruction of hands and manipulated objects},
  author={Hasson, Yana and Varol, Gul and Tzionas, Dimitrios and Kalevatykh, Igor and Black, Michael J and Laptev, Ivan and Schmid, Cordelia},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11807--11816},
  year={2019}
}
```

## [Baek 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Pushing_the_Envelope_for_RGB-Based_Dense_3D_Hand_Pose_Estimation_CVPR_2019_paper.pdf)

**Estimate hand mesh from RGB image**. The estimation method consists of three modules. First is a 2D evidence estimator which estimates 2D joints as heatmaps and foreground mask features. The foreground features and 2D joint skeletons are then sent to the 3D mesh estimator which consists of a 2D pose refiner and 3D mesh estimator net. The 3D mesh estimator net takes in it's previous output, refined 2D joint, and foreground feature and outputs the MANO parameters (shape and pose) and camera parameters (3D rotation in quaternion space, scale and translation). The 2D pose refiner takes in the previous 2D joint, previous 3D mesh estimator output and foreground feature and outputs a refined 2D pose. The estimates are refined iteratively. The final 3D mesh estimation output is passed to the projector which maps the hand mesh to its 3D pose and renders a foreground hand mask from the hand mesh. During testing, iterative refinement is carried out by both comparing the intermediate 2D joint pose with the 2D joint pose extracted from the hand mesh and the foreground feature  with the foreground hand mask rendered from the hand mesh. The paper also introduces a self-supervised data augmentation method which uses MANO and generates hand foreground mask, 3D hand pose, and using a neural texture renderer to render RGB.

```
@inproceedings{baek2019pushing,
  title={Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering},
  author={Baek, Seungryul and Kim, Kwang In and Kim, Tae-Kyun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1067--1076},
  year={2019}
}
```

## [Tekin 2019](https://arxiv.org/pdf/1904.05349)

**Multitask learning for 3D hand pose, 3D object pose, and action recognition from egocentric RGB image**. From egocentric RGB images, first the 3D hand pose, 3D object pose, confidence values for hand and object root, and object and action class for the frame are predicted by a fully convolutional network. The training is carried out in a supervised manner with a 3D grid-based learning structure. The network predicts each component for each cell in the pre-defined grid and chooses the prediction from the cell with the highest confidence. To incorporate long term dependencies, an LSTM module is utilized. Hand and object interaction are first modelled at the structured output level by modelling the depencies between them with a fully connected layer. The resulting output is then passed to the LSTM.

```
@inproceedings{tekin2019h+,
  title={H+ O: Unified egocentric recognition of 3D hand-object poses and interactions},
  author={Tekin, Bugra and Bogo, Federica and Pollefeys, Marc},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4511--4520},
  year={2019}
}
```

## [Yang 2019](https://arxiv.org/pdf/1812.01002.pdf)

**Learns disentangled representations of hand poses and hand images with a disentangled variational autoencoder**. Various training methods for specific senarios are introduced. The theory is applied on image synthesis and hand pose estimation from RGB images. For image synthesis, the latent variable is disentangled with respect to 3D pose and image (background) content (specified by a representative tag image). By varying the two variables, the synthesized image can be manipulated. In the case where representative tag images are hard to obtain (if each RGB image in the trianing set contains different background content), a different approach is introduced where the image content is indirectly modelled through the RGB image. For hand pose estimation, the latent variable is disentangled with respect to a canonical pose and viewpoint. Both of which can be used to obtain the 3D pose. The proposed method allows leveraging unlabelled or weakly-labelled data.

```
@inproceedings{yang2019disentangling,
  title={Disentangling Latent Hands for Image Synthesis and Pose Estimation},
  author={Yang, Linlin and Yao, Angela},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9877--9886},
  year={2019}
}
```

## [Zimmermann 2017](https://arxiv.org/pdf/1705.01389.pdf)

**3D RGB hand pose estimation**. 3 stages: Hand segmentation, 2D heatmap regression and finally estimating the 3D canonical pose and rotation matrix to transform the canonical pose to the final 3D hand pose.

```
@inproceedings{zimmermann2017learning,
  title={Learning to estimate 3d hand pose from single rgb images},
  author={Zimmermann, Christian and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4903--4911},
  year={2017}
}
```

## [Urooj 2018](https://arxiv.org/pdf/1803.03317.pdf)

**Analysis of hand segmentation**. Evaluates off-the-shelf methods on hand segmentation. Collects new datasets for hand segmentation in the wild.

```
@inproceedings{urooj2018analysis,
  title={Analysis of hand segmentation in the wild},
  author={Urooj, Aisha and Borji, Ali},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4710--4719},
  year={2018}
}
```

## [Cai 2019](https://cse.buffalo.edu/~jsyuan/papers/2019 Exploiting_Spatial-temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_Networks.pdf)

**3D pose estimation (hand and human) with graph-based method**. Estimate 3D pose from a short sequence of 2D joint locations. Domain specific knowledge is incorporated via a spatial-temporal graph. 

```
@article{cai2019exploiting,
  title={Exploiting spatial-temporal relationships for 3d pose estimation via graph convolutional networks},
  author={Cai, Yujun and Ge, Liuhao and Liu, Jun and Cai, Jianfei and Cham, Tat-Jen and Yuan, Junsong and Thalmann, Nadia Magnenat},
  year={2019}
}
```

## [Simon 2017](https://arxiv.org/pdf/1704.07809.pdf)

**2D hand pose estimation with multiview bootstrapping**. 2D hand pose estimation from RGB images. Refine by retraining model on similar poses but from more difficult viewpoints. Correct detections from easier viewpoints are used to triangulate 3D keypoints which can be reprojected to annotate viewpoints which the detector has failed. These new annotations can be used to retrain the system.

```
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

## [Chen 2019](https://drive.google.com/file/d/11GJzouV6jt_aOpvrJ8l3J5x_R_-m-Lg8/view)

**3D hand pose estimation from point cloud based on SO-Nets**. Semi-supervised training on both annotated and unannotated data. 

## [Xiong 2019](https://arxiv.org/pdf/1908.09999.pdf)

**Anchor based 3D pose estimation from depth images**. Anchor points are densely set up on depth images to capture global-local spatial context and predict joint positions in an ensemble manner. The network consists of 3 main branches, one for estimating the 2D joint offset from each anchor, the depth offset from each anchor and also the specific weighted response for each anchor to a certain joint. 

## Hand3D++: Hand Pose Estimation using Cascaded Pose-guided 3D Alignments

**3D hand pose estimation from 3D representation (point cloud or 3D volume)**. Estimation is carried out in a cascaded manner: global to palm to fingers. During the cascading, 3D alignments are performed to transform the 3D input to a joint specfic coordinate system. 

## [Zimmermann 2019](https://arxiv.org/pdf/1909.04349.pdf)

**Dataset for 3D hand pose and shape for single RGB images**. Carries out cross-dataset analysis on exisiting datasets and shows that they do perform well on the datasets they are trained on but not to other datasets. 

## Lapel-Net: Local-aware Permutation Equivariant Layer based 3D Hand Pose Estimation for Point Cloud

**3D hand pose estimation from point cloud**.

## [Narasimhaswamy 2019](https://arxiv.org/pdf/1904.04882.pdf)

**Hand detection and segmentation by extending MaskRCNN with a contextual attention mechanism**. Introduces large-scaled dataset for hands in unconstrained images.

```
@article{narasimhaswamy2019contextual,
  title={Contextual Attention for Hand Detection in the Wild},
  author={Narasimhaswamy, Supreeth and Wei, Zhengwei and Wang, Yang and Zhang, Justin and Hoai, Minh},
  journal={arXiv preprint arXiv:1904.04882},
  year={2019}
}
```
