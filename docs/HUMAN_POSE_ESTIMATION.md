
# Huamn Pose Estimation

## [Wang 2019](https://arxiv.org/pdf/1901.03798.pdf)

Bridges 2D to 3D domain gap with self-supervised correction mechanism that consists of 2D-to-3D pose transformation and 3D-to-2D pose projection.

```
@article{wang20193d,
  title={3D human pose machines with self-supervised learning},
  author={Wang, Keze and Lin, Liang and Jiang, Chenhan and Qian, Chen and Wei, Pengxu},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
  publisher={IEEE}
}
```

## [Wandt 2019](https://arxiv.org/pdf/1902.09868.pdf)

Tackles overfitting on 3D human pose estimation with an adversarial training method for with 2D projection and camera estimation

```
@inproceedings{wandt2019repnet,
  title={RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation},
  author={Wandt, Bastian and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7782--7791},
  year={2019}
}
```

## [Yang 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_3D_Human_Pose_CVPR_2018_paper.pdf)

 3D human pose estimation in the wild with adversarial learning to adapt structures learned from fully annotated 3D pose in constrained environments to 2D pose annotations in-the-wild

```
@inproceedings{yang20183d,
  title={3d human pose estimation in the wild by adversarial learning},
  author={Yang, Wei and Ouyang, Wanli and Wang, Xiaolong and Ren, Jimmy and Li, Hongsheng and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5255--5264},
  year={2018}
}
```

## [Mehta  2017](https://arxiv.org/pdf/1611.09813.pdf)

3D human body pose estimation from single RGB images by transfer learning from in-the-wild 2D pose annotations to constrained 3D pose annotations

```
@inproceedings{mehta2017monocular,
  title={Monocular 3d human pose estimation in the wild using improved cnn supervision},
  author={Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
  booktitle={2017 International Conference on 3D Vision (3DV)},
  pages={506--516},
  year={2017},
  organization={IEEE}
}
```

## [Zhou 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Towards_3D_Human_ICCV_2017_paper.pdf)

3D human pose estimation in-the-wild by jointly training on 2D and 3D datasets

```
@inproceedings{zhou2017towards,
  title={Towards 3d human pose estimation in the wild: a weakly-supervised approach},
  author={Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={398--407},
  year={2017}
}
```

## [Rogez 2016](http://papers.nips.cc/paper/6563-mocap-guided-data-augmentation-for-3d-pose-estimation-in-the-wild.pdf)

Synthesize images with 3D poses from 2D pose datasets by matching and stitching together 2D pose joints that match a 3D pose taken from a 3D pose dataset

```
@inproceedings{rogez2016mocap,
  title={Mocap-guided data augmentation for 3d pose estimation in the wild},
  author={Rogez, Gr{\'e}gory and Schmid, Cordelia},
  booktitle={Advances in neural information processing systems},
  pages={3108--3116},
  year={2016}
}
```

## [Jakab 2019](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/unsupervised_pose.pdf)

Image translation to estimate 2D human pose from unlabelled video frames and unpaired human keypoints.

```
@inproceedings{jakab2019image,
  title={Learning Human Pose from Unaligned Data through Image Translation},
  author={Jakab, Tomas and Gupta, Ankush and Bilen, Hakan and Vedaldi, Andrea},
}
```

## [Chen 2017](https://arxiv.org/pdf/1604.02703.pdf)

Improve 3D human pose estimation by joint training on synthesized and real training data with domain adaptation to map the synthetic and real data to the same domain.

```
@inproceedings{chen2016synthesizing,
  title={Synthesizing training images for boosting human 3d pose estimation},
  author={Chen, Wenzheng and Wang, Huan and Li, Yangyan and Su, Hao and Wang, Zhenhua and Tu, Changhe and Lischinski, Dani and Cohen-Or, Daniel and Chen, Baoquan},
  booktitle={2016 Fourth International Conference on 3D Vision (3DV)},
  pages={479--488},
  year={2016},
  organization={IEEE}
}
```

## [Wang 2019](https://arxiv.org/pdf/1903.07593.pdf)

Learn visual correspondence from video in a self-supervised manner using cycle consistency in time. Applies to video object segmentation, keypoint tracking, and optical flow.

```
@inproceedings{wang2019learning,
  title={Learning correspondence from the cycle-consistency of time},
  author={Wang, Xiaolong and Jabri, Allan and Efros, Alexei A},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2566--2576},
  year={2019}
}
```

## [???]()

Recurrent Pose Estimation Under Heavy Occlusion

**Detect multiple 2D human pose instances from one bounding box**. Extracts image and hidden features which are passed to a classifier to determine if there an instance in the image. If present, the features are used to estimate heatmaps. Hidden features are updated with the estimated heatmap and current hidden feature. This is to remove instance information if an instance is present. The entire process repeats until the classifier cannot detect anymore instances. A new dataset containing a wide variation of occlusion is synthesized by sampling images from the COCO keypoint dataset.

```
```