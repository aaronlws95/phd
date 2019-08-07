# Synthetic

## [Shrivastava 2017](https://arxiv.org/pdf/1612.07828.pdf)

**Improve the realism of synthetic images from unlabeled real data**. The method involves learning a refiner network that is trained in an adversarial manner to fool a discriminator that detects real or synthetic images. The training is also regularized by minimizing the image difference between sysnthetic and refined images. Local adversarial loss is used as any local patch sampled from the refined image should have similar statistics to a real image patch. Also, a lack of memory may cause divergence of training and the refiner network might re-introduce artifacts that the discriminator has forgotten about. To circumvent this, a history of refined images are used in training in addition to the current batch.

```
@inproceedings{shrivastava2017learning,
  title={Learning from simulated and unsupervised images through adversarial training},
  author={Shrivastava, Ashish and Pfister, Tomas and Tuzel, Oncel and Susskind, Joshua and Wang, Wenda and Webb, Russell},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2107--2116},
  year={2017}
}
```

## [Isola 2017](https://arxiv.org/pdf/1611.07004.pdf)

**Image to image translation with conditional generative adversarial networks**. The method is based on conditional GANs. For the generator, a U-net architecture is chosen with skip connections to circumvent the information bottleneck. For the discriminator, L1 loss is employed as it accurately captures low frequencies. For high frequencies, the attention is restricted to structures in local image patches (PatchGAN). 

```
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```