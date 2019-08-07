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