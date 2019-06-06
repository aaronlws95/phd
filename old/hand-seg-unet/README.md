# Hand Segmentation with U-net

Implementation of hand segmentation for learning purposes.

Experimental results [here](https://github.com/aaronlws95/hand-seg-unet/blob/master/EXPERIMENTS.md).

Setup instructions [here](https://github.com/aaronlws95/hand-seg-unet/blob/master/SETUP.md).

## Usage example

* Split the dataset and save for later usage:

`python data_split.py --dataset egohand`

* Convert data to TFRecords:

`python generate_tfrecord.py --dataset egohand`

* Training:

`python main.py --run_opt 1 --experiment unet --dataset egohand`

* Predicting:

`python main.py --run_opt 2 --experiment unet --dataset egohand --data_split train --load_epoch 100`

* Analysing predictions:

`python analse_predict.py --experiment unet --load_epoch 100 --dataset egohand --data_split train`

\*\*make sure to update directories in the files

## Resources

Guide on TFRecords: [https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36](https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36)

