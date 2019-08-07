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