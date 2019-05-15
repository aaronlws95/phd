# Datasets

## [First Person Hand Action (FPHA)](https://github.com/guiggh/hand_pose_action)

* RGB + Depth real hand images
    * Egocentric 
    * 106559 images
    * Images are frames from videos
    * 6 Subjects
    * RGB: 1920x1080
    * Depth: 640x480
    * Uses Intel RealSense SR300 
* 3D 21 joint hand annotation
    * obtained via magnetic sensors
* 6D Object pose annotation
    * 4 object types
    * obtained via magnetic sensors
* Object labels
    * 26 objects
* Action labels
    * 45 action categories
* Object models

## [Stereo Tracking Benchmark (STB)](https://arxiv.org/pdf/1610.07214.pdf)

* RGB + Depth real hand images
    * 3rd person 
    * 18000 images
    * Stereo RGB: 640 x 480
    * Depth: 640 x 480
    * 6 different backgrounds (3 static, 3 dynamic)
    * Stereo RGB captured from Point Grey Bumblebee2 stereo camera 
    * Depth captured from Intel RealSense f200
    * 2 types of hand poses (counting and random)
    * 12 video sequences with 1500 frames each
* 3D 21 joint hand annotation
    * manually labelled

## [Rendered Hand Dataset (RHD)](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)

* RGB + Depth synthetic hand images
    * 3rd person
    * 41258 training and 2728 testing samples
    * RGB: 320x320
    * Depth: 320x320
    * Created with Mixamo
    * Rendered with Blender
* Segmentation masks
     * 320x320
* 3D 21 joint hand annotation
    * xyz
    * uv
    * visibility indicator

## [Dexter + Object (D+O)](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm)

* RGB + Depth synthetic hand images
    * 3014 frames
    * 3rd person
    * 6 sequences
        * Grasp (2)
        * Pinch
        * Rigid
        * Rotate
        * Occlusion
    * 2 actors (1 female, 1 male)
    * RGB: Creative Senz3D color camera
    * Depth: Creative Senz3D close range TOF depth camera
* 3D fingertip positions
    * Manually annotated
* Object annotation
    * 3 object corners
* Depth camera intrinsics

## [Epic Kitchens (EK)](https://epic-kitchens.github.io/2018)

* RGB and optical flow images
    * 2 or 1 hand carrying out kitchen tasks
    * Egocentric
    * 11.5 million frames
    * Captured with head mounted camera
* Object bounding box
    * 454255 bounding boxes
    * Action 
* 39594 action segments
    * Narration
        * narrated sentence 
        * With timestamp
    * Labels
        * Verb and noun label
        * With start and end time of segment
        * 125 verb classes
        * 331 noun classes

## [Extended GTEA Gaze+ (EGTEA+)](http://www.cbi.gatech.edu/fpv/)

* 28 hours (de-identified) of cooking activities 
    * 86 unique sessions of 32 subjects.
* 10325 instances of fine-grained actions
    * e.g. “cut bell pepper”
* 15176 hand masks from 13,847 frames

