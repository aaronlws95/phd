# Datasets

## Sign Language Datasets

https://facundoq.github.io/unlp/sign_language_datasets/index.html

## Human Pose Estimation

| Dataset | Size | Dimension | Modality | Environment | Subjects per frame | Method |
|----------------------|----------------|-----------|----------|------------------------------------|--------------------|---------------------------------|
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php) | 3.6M | 3D | RGB | Lab | Single | Mocap |
| [MPII](http://human-pose.mpi-inf.mpg.de/#overview) | 25K | 2D | RGB | In-the-wild (Youtube) | Single/Multiple | Manual (AMT) |
| [MPI-INF3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) | 1.3M | 3D | RGB | Lab (Green screen) | Single | Mocap |
| [LSP](http://sam.johnson.io/research/lsp.html) | 2K | 2D | RGB | In-the-wild (Flickr Sports) | Single | Manual |
| [LSP extended](http://sam.johnson.io/research/lspet.html) | 10K | 2D | RGB | In-the-wild (Flickr Sports) | Single | Manual (AMT) |
| [HumanEva](http://humaneva.is.tue.mpg.de/) | 40K | 3D | RGB | Lab | Single | Mocap |
| [FLIC](https://bensapp.github.io/flic-dataset.html) | 5003 | 2D | RGB | Movies | Single | Manual (AMT) |
| [FLIC-plus](https://cims.nyu.edu/~tompson/flic_plus.htm) | 17380 | 2D | RGB | Movies | Single | Manual (AMT) |
| [Panoptic](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) | 11 hours | 3D | RGB | Lab (Panoptic studio) | Single/Multiple | [paper](https://arxiv.org/abs/1612.03153) |
| [3D Poses In The Wild](http://virtualhumans.mpi-inf.mpg.de/3DPW/) | 51K | 3D | RGB | In-the-wild | Single/Two | IMU |
| [COCO Keypoints](http://cocodataset.org/#keypoints-2018) | 200K | 2D | RGB | In-the-wild | Multiple | Manual (AMT) |
| [Unite the People](http://files.is.tuebingen.mpg.de/classner/up/) | 8128 | 3D | RGB | In-the-wild | Single | Fitting 2D keypoint to 3D model |
| [CMU Panoptic](http://domedb.perception.cs.cmu.edu/index.html) | 297K | 3D | RGB | Lab | Multiple | [paper](https://arxiv.org/pdf/1612.03153.pdf) |
| [CMU Mocap](http://mocap.cs.cmu.edu/) | 2500 sequences | 3D | RGB | Lab | Multiple | Mocap |
| [MPII Cooking](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-activities-dataset/) | 2348 | 2D | RGB | Kitchen | Single | Manual (Advene) |
| [JHMDB](http://jhmdb.is.tue.mpg.de/) | 33183 | 2D | RGB | In-the-wild (HMDB movie + Youtube) | Single | Manual (AMT + puppet tool) |

## Hand Pose Estimation

| Dataset | Size | Dimension | Modality | Environment | Hands per frame | Viewpoint | Method |
|----------------------------------|--------|-----------|----------------|-----------------------------------|-----------------|------------|------------------------------------|
| [FPHA](https://github.com/guiggh/hand_pose_action) | 106559 | 3D | RGB (noisy) +D | In-the-wild with objects | Single | Egocentric | Magnetic sensor |
| [STB](https://arxiv.org/pdf/1610.07214.pdf) | 18K | 3D | RGB+D | Lab | Single | 3rd Person | Manual |
| [RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) | 43986 | 3D | RGB+D | Synthetic | Single/Two | 3rd Person | Rendered (Mixamo, Blender) |
| [Dexter+Object](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm) | 3014 | 3D | RGB+D | Lab with objects | Single | 3rd Person | Manual |
| [CMU HandDB (manual)](http://domedb.perception.cs.cmu.edu/handdb.html) | 2758 | 2D | RGB | In-the-wild (MPII Youtube) + NZSL | Single | 3rd Person | Manual (fingertips) |
| [CMU HandDB (synthetic)](http://domedb.perception.cs.cmu.edu/handdb.html) | 14261 | 2D | RGB | Synthetic | Single | 3rd person | Rendered (Mixamo, Unreal Engine 4) |
| [CMU HandDB (multiview bootstrap)](http://domedb.perception.cs.cmu.edu/handdb.html) | 14817 | 2D | RGB | Lab (Panoptic Studio) | Single | 3rd person | [Multiview bootstrap](http://zpascal.net/cvpr2017/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf) |
| [BigHand2.2M](http://icvl.ee.ic.ac.uk/hands17/challenge/) | 2.2M | 3D | Depth | Lab (Depth) | Single | 3rd person | Magnetic sensor |
| [NYU](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) | 80K | 3D | Depth | Lab (Depth) | Single | 3rd person | [paper](https://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf) |
| [ICVL](https://labicvl.github.io/hand.html) | 331K | 3D | Depth | Lab (Depth) | Single | 3rd person | [Preliminary pose](https://arxiv.org/pdf/1705.07640.pdf) then refine |
| [MSRA15](https://github.com/geliuhao/CVPR2016_HandPoseEstimation/issues/4) | 76375 | 3D | Depth | Lab (Depth) | Single | 3rd person | [paper](http://www.jiansun.org/papers/CVPR14_HandTracking.pdf) |
| [SynthHands](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/SynthHands.htm) | 63540 | 3D | RGB+D | Synthetic | Single | Egocentric | Unity + LeapMotion |
| [EgoDexter](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm) | 1485 | 3D | RGB+D | Lab with objects | Single | Egocentric | Manual (fingertips) |
| [GANerated](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm) | 330K | 3D | RGB | Synthetic | Single | Egocentric | [paper](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/index.htm) |
| [HO-3D](https://arxiv.org/pdf/1907.01481.pdf) | 15K | 3D | RGB | Real | Single | 3rd Person | [Global optimization with 3D models and RGB+D camera]((https://arxiv.org/pdf/1907.01481.pdf)) |
| [Obman](https://www.di.ens.fr/willow/research/obman/data/) | 150K | 3D | RGB | Synthetic | Single | 3rd Person | [MANO, ShapeNet, GraspIt, SMPL](https://arxiv.org/pdf/1904.05767.pdf) |

## Hand Segmentation

| Dataset | Size | Modality | Environment | Hands per frame | Viewpoint | Method |
|----------------------------------|--------|-----------|----------------|-----------------------------------|-----------------|------------|
| [EYTH](https://github.com/aurooj/Hand-Segmentation-in-the-Wild) | 2600 | RGB | In-the-wild (Youtube) | Single/Two | Egocentric/3rd person | [LabelMe](https://github.com/wkentaro/labelme) |
| [Egohands](http://vision.soic.indiana.edu/projects/egohands/) | 4800 | RGB | In-the-wild | Single/Two | Egocentric/3rd person | Google Glass + Manual |
| [EGTEA](http://www.cbi.gatech.edu/fpv/) | 13847 | RGB | Kitchen | Single/Two | Egocentric | SMI eye-tracking glasses + BasicFinder |