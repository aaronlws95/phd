# Generation

# [Siarohin, 2018](https://arxiv.org/pdf/1801.00055.pdf)

**Generating person images conditioned on a given pose given an image of a person and a target pose**. The method employs GANs with a U-Net based generator and discriminator based on Pix2Pix. Training input is a pair of images of the same person with different poses. The authors also introduce a deformable skip connections which deforms the feature maps of the encoder with affine transformations of each specific body part pose for the pair of images before concatenating the feature with the corresponding feature in the decoder. A nearest neighbour loss is used over conventional L1 or L2 losses.

```
@inproceedings{siarohin2018deformable,
  title={Deformable gans for pose-based human image generation},
  author={Siarohin, Aliaksandr and Sangineto, Enver and Lathuili{\`e}re, St{\'e}phane and Sebe, Nicu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3408--3416},
  year={2018}
}
```

# [Ma, 2018](https://arxiv.org/pdf/1705.09368.pdf)

**Synthesis of person images in arbitrary poses based on an image of that person and a novel pose**. The method employs GANs with a U-net based generator. It consists of 2 stages. Stage-I consists of pose integration. The idea is to take the conditioning image and the target pose to generate a course intermediate result that captures the global structure of the human body in the target image. Stage-II consists of image refinement. The model now focuses on generating more refined details using a variant of conditional DCGAN.

```
@inproceedings{ma2017pose,
  title={Pose guided person image generation},
  author={Ma, Liqian and Jia, Xu and Sun, Qianru and Schiele, Bernt and Tuytelaars, Tinne and Van Gool, Luc},
  booktitle={Advances in Neural Information Processing Systems},
  pages={406--416},
  year={2017}
}
```