# Self Supervised

## [Oord 2018](https://arxiv.org/pdf/1807.03748.pdf)

**Unsupervised learning for learning representations from high-dimensional data**. Contrastive Predictive Coding (CPC) learns representations by predicting the future in latent space using autoregressive models. The method uses a probabilitic contrastive loss based on Noise Constrastive Estimation (NCE) which induces the latent space to capture information that is maximally useful to predict future samples. The authors show that this technique is applicable to speech, images, text and reinforcement learning in 3D environments.

```
@article{oord2018representation,
  title={Representation learning with contrastive predictive coding},
  author={Oord, Aaron van den and Li, Yazhe and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1807.03748},
  year={2018}
}
```

## [Tian 2019](https://arxiv.org/pdf/1906.05849.pdf)

**Contrastive learning in a multiview setting**. Authors introduce a method to learn representations from multiview data. A constrastive objective is utilized to encourage congruent views to be brought together in representation space. The latent encoding for each view  is concatenated to form the full representation of a scene. For more than two views, a Core View (one view and all other views contrasted against that view) and Full Graph (contrast every pair and representation for all views are jointly learned) scheme is proposed. An NCE-based contrastive loss (different to CPC) is used. The authors also introduce an alternative patch-based contrastive loss that performs worst empirically but employed when the dataset is small.

```
@article{tian2019contrastive,
  title={Contrastive Multiview Coding},
  author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  journal={arXiv preprint arXiv:1906.05849},
  year={2019}
}
```

## [Unsupervised Learning of Visual Representations using Videos](https://arxiv.org/pdf/1505.00687.pdf)

**Self-supervised learning by tracking patches**. Uses Siamese Triplet Network and triplet loss to learn from image frames extracted from Youtube. The network is first trained on randomly sampled negatives and then on harder negatives with the highest loss in a given batch. SURF is used to extract points from the video frames. The network is fine-tuned on object recognition and surface normal estimation.

```
@inproceedings{wang2015unsupervised,
  title={Unsupervised learning of visual representations using videos},
  author={Wang, Xiaolong and Gupta, Abhinav},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2794--2802},
  year={2015}
}
```

## [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379.pdf)

**Feature learning by inpainting**. Use inpainting random regions with reconstruction and adversarial loss on images to learn features. Network is an encoder decoder. Results are evaluated on inpainting and also using the encoder and fine-tuning on classification, detection and semantic segmentation tasks.

```
@inproceedings{pathak2016context,
  title={Context encoders: Feature learning by inpainting},
  author={Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2536--2544},
  year={2016}
}
```