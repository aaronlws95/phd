# Hand Pose Estimation

## [Wan 2019](http://www.vision.ee.ethz.ch/~wanc/papers/cvpr2019.pdf)

Self-supervised 3D hand pose estimation from depth maps using a spherical hand model and multi-view supervision.
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

3D hand pose estimation from depth by feature mapping real image features to synthetic features and joint training on real and synthetic images.
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

Joint training on 3D and 2D hand annotations with reprojection to predict 3D hand shape (mesh) and pose from RGB.
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

Weak supervision of 3D hand pose estimation from RGB by generating depth maps from predicted 3D poses.
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

Translate synthetic images to real using GAN for 3D hand pose estimation from RGB.

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