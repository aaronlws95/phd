# Audio Animation

## [Ginosar 2019](https://arxiv.org/pdf/1906.04160.pdf)

**Gesture generation from speech audio**. Generate own dataset of videos with pseudo ground-truth labels annotated by OpenPose. Proposed method is fully convolutional and maps a 2D log-mel spectrogram to a temporal stack of 2D pose vectors. An additional adversarial discriminator is added to ensure plausible motion.

```
@inproceedings{ginosar2019learning,
  title={Learning Individual Styles of Conversational Gesture},
  author={Ginosar, Shiry and Bar, Amir and Kohavi, Gefen and Chan, Caroline and Owens, Andrew and Malik, Jitendra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3497--3506},
  year={2019}
}
```

## [Chan 2018](https://arxiv.org/pdf/1808.07371.pdf)

**Transfer motion between two human subjects**. Uses pose as an intermediate representation. During inference, global pose normalization is implemented to ensure the pose is consistent with the poses from the target video. Temporal smoothing is added to enforce temporal coherence and a FaceGAN is added to add detail and realism to the face region.

```
@article{chan2018everybody,
  title={Everybody dance now},
  author={Chan, Caroline and Ginosar, Shiry and Zhou, Tinghui and Efros, Alexei A},
  journal={arXiv preprint arXiv:1808.07371},
  year={2018}
}
```

## [Schlizerman 2017](https://arxiv.org/pdf/1712.09382.pdf)

**Learns pose predictions of a person playing violin or piano given audio input**. Data is collected using OpenPose, MaskRCNN and DeepFace. The method uses unidirectional LSTM to estimate the PCA components (representing pose prediction) from MFCC coefficient input (representing the audio). They also use  the estimated keypoints to animate and AR avatar.

```
@inproceedings{shlizerman2018audio,
  title={Audio to body dynamics},
  author={Shlizerman, Eli and Dery, Lucio and Schoen, Hayden and Kemelmacher-Shlizerman, Ira},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7574--7583},
  year={2018}
}
```

## [Tian 2019](https://arxiv.org/pdf/1905.11142.pdf)

**Generating facial animations from audio input only**. Use bidirectional LSTM with an attention layer to map MFCC coefficients to abstract representations which are used to generate 51-dim vectors to control a 3D face model. Training data obtained using commercial Faceshift. Loss consists of a target loss and a smooth loss.

```
@article{tian2019audio2face,
  title={Audio2Face: Generating Speech/Face Animation from Single Audio with Attention-Based Bidirectional LSTM Networks},
  author={Tian, Guanzhong and Yuan, Yi and others},
  journal={arXiv preprint arXiv:1905.11142},
  year={2019}
}
```

## [Ding 2017](https://arxiv.org/pdf/1709.03842.pdf)

**Controllable transformation of expression from a given face to a target face**. Encoder maps input image to a latent representation, an expression controller module converts target expression labels to an expressive code. The latent representation and expressive code are used as input to the decoder to generate a reconstructed image. Discriminators are used to refine the image for photo-realistic textures and to ensure the learned identity representation is filled. Network is trained incrementally. Expression intensities can be continuously adjusted.

```
@inproceedings{ding2018exprgan,
  title={Exprgan: Facial expression editing with controllable expression intensity},
  author={Ding, Hui and Sricharan, Kumar and Chellappa, Rama},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

## [Chen 2019](https://arxiv.org/pdf/1905.03820.pdf)

**Generate talking face from audio**. GAN-based cascade network structure. AT-Net takes audio (MFCC) and facial landmarks (PCA) as input and outputs facial landmarks (PCA) corresponding to the input audio. VG-Net takes the original input and the output PCA from the AT-Net and generates a new face. Authors also introduces an attention-base dynamic pixel loss to enforce generation of temporally consistent pixels and a regression-based discriminator.

```
@article{chen2019hierarchical,
  title={Hierarchical Cross-Modal Talking Face Generationwith Dynamic Pixel-Wise Loss},
  author={Chen, Lele and Maddox, Ross K and Duan, Zhiyao and Xu, Chenliang},
  journal={arXiv preprint arXiv:1905.03820},
  year={2019}
}
```

## [Chen 2018](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0003046.pdf)

**Generate facial animation from audio and video information**. (Audio) Large vocabulary speech recognition (LVCSR) is employed to obtain a sequence of phonemes from audio input. These phonemes are mapped to expression blendshapes. (Video) 2D face landmarks are detected using an off-the-shelf method which is used to reconstruct a 3D face model.

```
@inproceedings{chen2018generate,
author={X. {Chen} and C. {Cao} and Z. {Xue} and W. {Chu}},
booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
title={Joint Audio-Video Driven Facial Animation},
year={2018},
pages={3046-3050},
doi={10.1109/ICASSP.2018.8461502},
ISSN={2379-190X},
month={April},}
```

## [Kim 2019](https://arxiv.org/pdf/1907.02253.pdf)

**Generate full-pose lecture videos from instructor audio narration of any length**. A bidirectional LSTM extracts latent codes given audio input (log Mel-filterbank energy features). Then, a VAE decoder constructs corresponding pose figures (DensePose) from the latent code. Finally, the corresponding pose figure is translated to generate final video frames using a GAN-model based on Pix2Pix.

```
@article{kim2019lumi,
  title={Lumi$\backslash$ereNet: Lecture Video Synthesis from Audio},
  author={Kim, Byung-Hak and Ganapathi, Varun},
  journal={arXiv preprint arXiv:1907.02253},
  year={2019}
}
```

## [Wen 2019](https://arxiv.org/pdf/1905.10604.pdf)

**Reconstructing face from voice**. GAN-based network with a voice embedding network that embeds the audio input (Mel-Spectrographic) and a generator which generates a face from the embedding. The network comes with two adversaries, one to determine if the input is a real image and one to assign an identity label to a face.

```
@article{DBLP:journals/corr/abs-1905-10604,
  author    = {Yandong Wen and
               Rita Singh and
               Bhiksha Raj},
  title     = {Reconstructing faces from voices},
  journal   = {CoRR},
  volume    = {abs/1905.10604},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.10604},
  archivePrefix = {arXiv},
  eprint    = {1905.10604},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-10604},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## [Oh 2019](https://arxiv.org/pdf/1905.09773.pdf)

**Reconstructing face from short audio recording**. Training is done in a self-supervised manner. A voice encoder network converts the short audio spectrogram input to a face feature vector. The feature vector is passed to a face decoder (pre-trained) which reconstructs the image of a face from the face feature. During training, face features extracted from a face recognition model is used  in loss calculations.

```
@inproceedings{oh2019speech2face,
  title={Speech2Face: Learning the Face Behind a Voice},
  author={Oh, Tae-Hyun and Dekel, Tali and Kim, Changil and Mosseri, Inbar and Freeman, William T and Rubinstein, Michael and Matusik, Wojciech},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7539--7548},
  year={2019}
}
```

## [Liu 2015](http://cgcad.thss.tsinghua.edu.cn/xufeng/cameraReady_sigAsia15.pdf)

**Generate facial animation from audio and video information**. In training stage, train a DNN to predict phoneme state posterior predictions (PSPP) from audio input (single-sided contextual features). An audio-visual database is constructed using the DNN to extract PSPP (audio) and a multi-linear model with Mocap data is used to track facial motion from visual input (video). In the online stage, the multi-linear model is used to extract visual features and the DNN to extract audio features. Together with the database, the features are used to reconstruct the mouth shape. The final output is composed from the initial tracked expression and the final mouth shape.

```
@article{liu2015video,
  title={Video-audio driven real-time facial animation},
  author={Liu, Yilong and Xu, Feng and Chai, Jinxiang and Tong, Xin and Wang, Lijuan and Huo, Qiang},
  journal={ACM Transactions on Graphics (TOG)},
  volume={34},
  number={6},
  pages={182},
  year={2015},
  publisher={ACM}
}
```

## [Doukas 2019](https://arxiv.org/pdf/1905.12043.pdf)

**Transform an input video of a spoken word to an output video of a different word**. Method builds on StarGAN. Input is a video and a target world label. The generator transforms the input into a video with the target uttering the target word. A discriminator is implemented to distinguish the real video from the fake video. Additionally, to ensure the target word is uttered in the video, a word classifier and character inspector is added. The original video is also reconstructed from the generated video to preserve speaker identity. There is also a feature matching loss which matches features from the fake video with a real video uttering a similar word.

```
@article{doukas2019video,
  title={Video-to-Video Translation for Visual Speech Synthesis},
  author={Doukas, Michail C and Sharmanska, Viktoriia and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1905.12043},
  year={2019}
}
```

## [Kopuklu 2019](https://arxiv.org/pdf/1905.04225.pdf)

**Detect sequence of hand gestures**. Use 2D/3D CNN Classifier based on SqueezeNet or MobileNet to classify frames obtained from sliding windows on a video. Windows are processed sequentially with cues from starting gestures and ending gestures to control activation. Finally, a viterbi-like decoder runs on the classifier queue to recognize the gesture sequence. A custom dataset of IR/Depth images of 13/15 class gestures with both dynamic and static gestures are collected.

```
@article{kopuklu2019talking,
  title={Talking with Your Hands: Scaling Hand Gestures and Recognition with CNNs},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Rong, Yao and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:1905.04225},
  year={2019}
}
```

## [Suwajanakorn 2017](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)

**Synthesize videos of Obama speaking from audio input of Obama speaking**. Method involves applying a unidirectional LSTM to map audio input (MFCC) to mouth shape features (PCA). Given the mouth features, best matching frames are selected from the video and  weighted median synthesis with proxy-based teeth enhancement is applied to synthesize lower face texture. Dynamic programming is used to re-time target videos to align audio and visual pauses. The face texture and re-timed target video is then composited together by Laplacian pyramid blending with jaw correction.

```
@article{suwajanakorn2017synthesizing,
  title={Synthesizing obama: learning lip sync from audio},
  author={Suwajanakorn, Supasorn and Seitz, Steven M and Kemelmacher-Shlizerman, Ira},
  journal={ACM Transactions on Graphics (TOG)},
  volume={36},
  number={4},
  pages={95},
  year={2017},
  publisher={ACM}
}
```

## [Camgoz 2018](https://www-i6.informatik.rwth-aachen.de/publications/download/1064/CamgozCihanHadfieldSimonKollerOscarNeyHermannBowdenRichard--NeuralSignLanguageTranslation--2018.pdf)

**Sign language translation from video**. Network contains usage of CNN to learn spatial embedding from image frames and an attention-based encoder-decoder to map from tokenized (frame or gloss level) input to the target sentence.

```
@inproceedings{cihan2018neural,
  title={Neural sign language translation},
  author={Cihan Camgoz, Necati and Hadfield, Simon and Koller, Oscar and Ney, Hermann and Bowden, Richard},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7784--7793},
  year={2018}
}
```

## [Koller 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Koller_Deep_Hand_How_CVPR_2016_paper.pdf)

**Hand shape classification from sign language videos**. Using weakly labelled sequence date, the method embeds a CNN within an iterative EM algorithm.

```
@inproceedings{koller2016deep,
  title={Deep hand: How to train a CNN on 1 million hand images when your data is continuous and weakly labelled},
  author={Koller, Oscar and Ney, Hermann and Bowden, Richard},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3793--3802},
  year={2016}
}
```

## [Ko 2019](https://arxiv.org/pdf/1811.11436.pdf)

**Sign language translation from video with human keypoint estimation**. Create new dataset for Korean Sign Language. Uses seq2seq based translation method with an attention-based encoder-decoder network based on RNN. The method uses human keypoint estimation features and feature normalization for its input.

```
@article{ko2019neural,
  title={Neural Sign Language Translation based on Human Keypoint Estimation},
  author={Ko, Sang-Ki and Kim, Chang Jo and Jung, Hyedong and Cho, Choongsang},
  journal={Applied Sciences},
  volume={9},
  number={13},
  pages={2683},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## [Zhou 2019](https://arxiv.org/pdf/1807.07860.pdf)

**Generates talking face given an arbitrary face image and either input audio or video of speech**. The method involves learning a disentangled audio-visual representation between Person-ID space (pid) and Word-ID space (wid) in an adversarial manner. The audio-visual representation is learned such that audio and visual features are indistinguishable. Thus, a face generator can be learned by combining the disentangled pid feature and wid feature (either audio or video).

```P
@inproceedings{zhou2019talking,
  title={Talking face generation by adversarially disentangled audio-visual representation},
  author={Zhou, Hang and Liu, Yu and Liu, Ziwei and Luo, Ping and Wang, Xiaogang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={9299--9306},
  year={2019}
}
```