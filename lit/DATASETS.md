# Datasets

## Sign Language Datasets

https://facundoq.github.io/unlp/sign_language_datasets/index.html

## Human Pose Estimation

| Dataset | Size | Dimension | Modality | Environment | Subjects per frame | Method |
|----------------------|----------------|-----------|----------|------------------------------------|--------------------|---------------------------------|
| Human3.6M | 3.6M | 3D | RGB | Lab | Single | Mocap |
| MPII | 25K | 2D | RGB | In-the-wild (Youtube) | Single/Multiple | Manual (AMT) |
| MPI-INF3DHP | 1.3M | 3D | RGB | Lab (Green screen) | Single | Mocap |
| LSP | 2K | 2D | RGB | In-the-wild (Flickr Sports) | Single | Manual |
| LSP extended | 10K | 2D | RGB | In-the-wild (Flickr Sports) | Single | Manual (AMT) |
| HumanEva | 40K | 3D | RGB | Lab | Single | Mocap |
| FLIC | 5003 | 2D | RGB | Movies | Single | Manual (AMT) |
| FLIC-plus | 17380 | 2D | RGB | Movies | Single | Manual (AMT) |
| Panoptic | 11 hours | 3D | RGB | Lab (Panoptic studio) | Single/Multiple | paper |
| 3D Poses In The Wild | 51K | 3D | RGB | In-the-wild | Single/Two | IMU |
| COCO Keypoints | 200K | 2D | RGB | In-the-wild | Multiple | Manual (AMT) |
| Unite the People | 8128 | 3D | RGB | In-the-wild | Single | Fitting 2D keypoint to 3D model |
| CMU Panoptic | 297K | 3D | RGB | Lab | Multiple | paper |
| CMU Mocap | 2500 sequences | 3D | RGB | Lab | Multiple | Mocap |
| MPII Cooking | 2348 | 2D | RGB | Kitchen | Single | Manual (Advene) |
| JHMDB | 33183 | 2D | RGB | In-the-wild (HMDB movie + Youtube) | Single | Manual (AMT + puppet tool) |

## Hand Pose Estimation

| Dataset | Size | Dimension | Modality | Environment | Hands per frame | Viewpoint | Method |
|----------------------------------|--------|-----------|----------------|-----------------------------------|-----------------|------------|------------------------------------|
| FPHA | 106559 | 3D | RGB (noisy) +D | In-the-wild with objects | Single | Egocentric | Magnetic sensor |
| STB | 18K | 3D | RGB+D | Lab | Single | 3rd Person | Manual |
| RHD | 43986 | 3D | RGB+D | Synthetic | Single/Two | 3rd Person | Rendered (Mixamo, Blender) |
| Dexter+Object | 3014 | 3D | RGB+D | Lab with objects | Single | 3rd Person | Manual |
| CMU HandDB (manual) | 2758 | 2D | RGB | In-the-wild (MPII Youtube) + NZSL | Single | 3rd Person | Manual (fingertips) |
| CMU HandDB (synthetic) | 14261 | 2D | RGB | Synthetic | Single | 3rd person | Rendered (Mixamo, Unreal Engine 4) |
| CMU HandDB (multiview bootstrap) | 14817 | 2D | RGB | Lab (Panoptic Studio) | Single | 3rd person | Multiview bootstrap |
| BigHand2.2M | 2.2M | 3D | Depth | Lab (Depth) | Single | 3rd person | Magnetic sensor |
| NYU | 80K | 3D | Depth | Lab (Depth) | Single | 3rd person | paper |
| ICVL | 331K | 3D | Depth | Lab (Depth) | Single | 3rd person | Preliminary pose then refine |
| MSRA15 | 76375 | 3D | Depth | Lab (Depth) | Single | 3rd person | paper |
| SynthHands | 63540 | 3D | RGB+D | Synthetic | Single | Egocentric | Unity + LeapMotion |
| EgoDexter | 1485 | 3D | RGB+D | Lab with objects | Single | Egocentric | Manual (fingertips) |
| GANerated | 330K | 3D | RGB | Synthetic | Single | Egocentric | paper |