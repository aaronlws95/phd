# Sign Language

## [Yang 2019](https://arxiv.org/pdf/1908.01341.pdf)

**Continuous sign language recognition from multiple levels of semantic information**. The paper introduces Structured Feature Network (SF-Net) which extracts features in a structure manner from sign lagnuage videos. Information is processed at 3 different levels. The first level is the frame level which incorporates 2D and 3D convolution to capture gesture, emotion and fast and small motions. The second level is the gloss level which generates meta frames which are processed by an LSTM to learn long term motions. Finally, the features are processed by a Bi-LSTM in the sentence level to encode context information. Sign languages: Chinese Sign Language, German Sign Language

```
@article{yang2019sf,
  title={SF-Net: Structured Feature Network for Continuous Sign Language Recognition},
  author={Yang, Zhaoyang and Shi, Zhenmei and Shen, Xiaoyong and Tai, Yu-Wing},
  journal={arXiv preprint arXiv:1908.01341},
  year={2019}
}
```

## [Stoll 2018](http://bmvc2018.org/contents/papers/0906.pdf)

**Generates sign language frames from spoken language sentences**. The method involves 3 stages: neural machine translation from spoken language to sign gloss, mapping sign gloss to corresponding skeleton pose with a lookup table, and finally generating the sign language frames given a base input image and the skeleton pose sequence. The skeleton pose consists of 10 keypoints corresponding to the upper body which do not include detailed hand keypoints. Sign languages: German Sign Language, Swiss German Sign Language

```
@inproceedings{stoll2018sign,
  title={Sign Language Production using Neural Machine Translation and Generative Adversarial Networks.},
  author={Stoll, Stephanie and Hadfield, Simon and Bowden, Richard}
}
```

## [Camgoz 2018](https://www-i6.informatik.rwth-aachen.de/publications/download/1064/CamgozCihanHadfieldSimonKollerOscarNeyHermannBowdenRichard--NeuralSignLanguageTranslation--2018.pdf)

**Sign language translation from video**. Network contains usage of CNN to learn spatial embedding from image frames and an attention-based encoder-decoder to map from tokenized (frame or gloss level) input to the target sentence. Sign languages: German Sign Language

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

**Hand shape classification from sign language videos**. Using weakly labelled sequence date, the method embeds a CNN within an iterative EM algorithm. Sign languages: German Sign Language

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

**Sign language translation from video with human keypoint estimation**. Create new dataset for Korean Sign Language. Uses seq2seq based translation method with an attention-based encoder-decoder network based on RNN. The method uses human keypoint estimation features and feature normalization for its input. Sign languages: Korean Sign Language

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

# [Cox, 2002](http://www.cstr.ed.ac.uk/downloads/publications/2002/Cox-Assets-2000.pdf)

**System that translates speech into sign language and displays it using an avatar in a Post Office setting**. The system consists of a speech recogniser (adapted to each individual speaker) which is constrained to a finite number of predefined phrases that is decided by analysing transcripts of Post Office transactions. The phrases are then synthesized into the corresponding sign by an avatar, TESSA. The signs are collected by using Cybergloves, Polhemus magnetic sensor and a head mounted camera with infra-red filters. Sign languages: British Sign Language

"For many people who have been profoundly deaf from a
young age, signing is their first language so they learn to
read and write English as a second language. As a
result, many deaf people have below-average reading
abilities for English text and prefer to communicate using
sign language"

"Two variants of sign language are possible for
communication with deaf people in the UK: British Sign
Language (BSL) and Sign-supported English (SSE). BSL
is a fully developed language with it own syntactical and
semantic structures, whereas SSE uses the same (or
very similar) signs for words as BSL, but uses English
language word order. Using pre-stored SSE “words”
would enable sentences to be translated into sign
language, but SSE is not popular with the deaf
community and it is very important that the system is
acceptable to deaf users."

```
@inproceedings{cox2002tessa,
  title={Tessa, a system to aid communication with deaf people},
  author={Cox, Stephen and Lincoln, Michael and Tryggvason, Judy and Nakisa, Melanie and Wells, Mark and Tutt, Marcus and Abbott, Sanja},
  booktitle={Proceedings of the fifth international ACM conference on Assistive technologies},
  pages={205--212},
  year={2002},
  organization={ACM}
}
```

# [Glauert, 2006](http://www2.cmp.uea.ac.uk/~sjc/Technology+Disability-2007.pdf)

**System that translates speech or text into sign language and displays it using an avatar for specific settings e.g. eGovernment**. Introduce VANESSA, development follows from TESSA.

```
@article{glauert2006vanessa,
  title={VANESSA--A system for communication between Deaf and hearing people},
  author={Glauert, JRW and Elliott, R and Cox, SJ and Tryggvason, J and Sheard, M},
  journal={Technology and Disability},
  volume={18},
  number={4},
  pages={207--216},
  year={2006},
  publisher={IOS Press}
}
```

# [McDonald 2016](https://scholarworks.bgsu.edu/cgi/viewcontent.cgi?article=1032&context=vcte_pub)

**Add realism and optimization to sign language animation from a sparse set of key frames**.

```
@article{mcdonald2016automated,
  title={An automated technique for real-time production of lifelike animations of American Sign Language},
  author={McDonald, John and Wolfe, Rosalee and Schnepp, Jerry and Hochgesang, Julie and Jamrozik, Diana Gorman and Stumbo, Marie and Berke, Larwan and Bialek, Melissa and Thomas, Farah},
  journal={Universal Access in the Information Society},
  volume={15},
  number={4},
  pages={551--566},
  year={2016},
  publisher={Springer}
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

# SignAll

**ASL to English translation**. System is installed in a fixed position. Requires gloves to detect hand points.

```
@online{signall,
  author = {Zsolt Robotka},
  url = {https://www.signall.us/},
}
```

# KinTrans

**Real-time, multi-lingual sign language translator**.

```
@online{kintrans,
  author = {Mohamed Elwazer},
  url = {https://www.kintrans.com/},
}
```
