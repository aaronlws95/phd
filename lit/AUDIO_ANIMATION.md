# Audio Animation

## [Ginosar 2019](https://arxiv.org/pdf/1906.04160.pdf)

*Gesture generation from speech audio*. Generate own dataset of videos with pseudo ground-truth labels annotated by OpenPose. Proposed method is fully convolutional and maps a 2D log-mel spectrogram to a temporal stack of 2D pose vectors. An additional adversarial discriminator is added to ensure plausible motion.

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

*Transfer motion between two human subjects*. Uses pose as an intermediate representation. During inference, global pose normalization is implemented to ensure the pose is consistent with the poses from the target video. Temporal smoothing is added to enforce temporal coherence and a FaceGAN is added to add detail and realism to the face region.

## [Schlizerman 2017](https://arxiv.org/pdf/1712.09382.pdf)

*Learns pose predictions of a person playing violin or piano given audio input*. Data is collected using OpenPose, MaskRCNN and DeepFace. The method uses unidirectional LSTM to estimate the PCA components (representing pose prediction) from MFCC coefficient input (representing the audio). They also use  the estimated keypoints to animate and AR avatar.

## [Tian 2019](https://arxiv.org/pdf/1905.11142.pdf)

*Generating facial animations from audio input only*. Use bidirectional LSTM with an attention layer to map MFCC coefficients to abstract representations which are used to generate 51-dim vectors to control a 3D face model. Training data obtained using commercial Faceshift. Loss consists of a target loss and a smooth loss.

## [Ding 2017](https://arxiv.org/pdf/1709.03842.pdf)

*Controllable transformation of expression from a given face to a target face*. Encoder maps input image to a latent representation, an expression controller module converts target expression labels to an expressive code. The latent representation and expressive code are used as input to the decoder to generate a reconstructed image. Discriminators are used to refine the image for photo-realistic textures and to ensure the learned identity representation is filled. Network is trained incrementally. Expression intensities can be continuously adjusted.

## [Chen 2019](https://arxiv.org/pdf/1905.03820.pdf)

*Generate talking face from audio*. GAN-based cascade network structure. AT-Net takes audio (MFCC) and facial landmarks (PCA) as input and outputs facial landmarks (PCA) corresponding to the input audio. VG-Net takes the original input and the output PCA from the AT-Net and generates a new face. Authors also introduces an attention-base dynamic pixel loss to enforce generation of temporally consistent pixels and a regression-based discriminator.

## [Chen 2018](https://ieeexplore.ieee.org/document/8461502)

*Generate facial animation from audio and video information*. (Audio) Large vocabulary speech recognition (LVCSR) is employed to obtain a sequence of phonemes from audio input. These phonemes are mapped to expression blendshapes. (Video) 2D face landmarks are detected using an off-the-shelf method which is used to reconstruct a 3D face model.

## [Kim 2019](https://arxiv.org/pdf/1907.02253.pdf)

*Generate full-pose lecture videos from instructor audio narration of any length*. A bidirectional LSTM extracts latent codes given audio input (log Mel-filterbank energy features). Then, a VAE decoder constructs corresponding pose figures (DensePose) from the latent code. Finally, the corresponding pose figure is translated to generate final video frames using a GAN-model based on Pix2Pix.

## [Wen 2019](https://arxiv.org/pdf/1905.10604.pdf)

*Reconstructing face from voice*. GAN-based network with a voice embedding network that embeds the audio input (Mel-Spectrographic) and a generator which generates a face from the embedding. The network comes with two adversaries, one to determine if the input is a real image and one to assign an identity label to a face.

## [Oh 2019](https://arxiv.org/pdf/1905.09773.pdf)

*Reconstructing face from short audio recording*. Training is done in a self-supervised manner. A voice encoder network converts the short audio spectrogram input to a face feature vector. The feature vector is passed to a face decoder (pre-trained) which reconstructs the image of a face from the face feature. During training, face features extracted from a face recognition model is used  in loss calculations.

## [Liu 2015](http://cgcad.thss.tsinghua.edu.cn/xufeng/cameraReady_sigAsia15.pdf)

*Generate facial animation from audio and video information*. In training stage, train a DNN to predict phoneme state posterior predictions (PSPP) from audio input (single-sided contextual features). An audio-visual database is constructed using the DNN to extract PSPP (audio) and a multi-linear model with Mocap data is used to track facial motion from visual input (video). In the online stage, the multi-linear model is used to extract visual features and the DNN to extract audio features. Together with the database, the features are used to reconstruct the mouth shape. The final output is composed from the initial tracked expression and the final mouth shape.

## [Doukas 2019](https://arxiv.org/pdf/1905.12043.pdf)

*Transform an input video of a spoken word to an output video of a different word*. Method builds on StarGAN. Input is a video and a target world label. The generator transforms the input into a video with the target uttering the target word. A discriminator is implemented to distinguish the real video from the fake video. Additionally, to ensure the target word is uttered in the video, a word classifier and character inspector is added. The original video is also reconstructed from the generated video to preserve speaker identity. There is also a feature matching loss which matches features from the fake video with a real video uttering a similar word.

## [Kopuklu 2019](https://arxiv.org/pdf/1905.04225.pdf)

*Detect sequence of hand gestures*. Use 2D/3D CNN Classifier based on SqueezeNet or MobileNet to classify frames obtained from sliding windows on a video. Windows are processed sequentially with cues from starting gestures and ending gestures to control activation. Finally, a viterbi-like decoder runs on the classifier queue to recognize the gesture sequence. A custom dataset of IR/Depth images of 13/15 class gestures with both dynamic and static gestures are collected.

## [Suwajanakorn 2017](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)

*Synthesize videos of Obama speaking from audio input of Obama speaking*. Method involves applying a unidirectional LSTM to map audio input (MFCC) to mouth shape features (PCA). Given the mouth features, best matching frames are selected from the video and  weighted median synthesis with proxy-based teeth enhancement is applied to synthesize lower face texture. Dynamic programming is used to re-time target videos to align audio and visual pauses. The face texture and re-timed target video is then composited together by Laplacian pyramid blending with jaw correction.

## [Hasson 2019](https://arxiv.org/pdf/1904.05767.pdf)

*Joint reconstruction of object and hand mesh from monocular RGB*. Method employs two main branches, one for hand and one for object. For the hand branch, the MANO model (PCA) which maps a pose and shape parameter to a mesh is integrated as a differentiable layer. For the object branch, Atlasnet is employed for object prediction. Scale and translation is also predicted to keep object position and scale relative to the hand. Repulsion loss is introduced  to penalize interpentration and an attraction loss is introduced to penalize cases where the surfaces are not in contact.