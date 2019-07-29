# action-recognition-LSTM
Experiments with LSTM on action recognition

Using dataset obtained from [here](https://github.com/guiggh/hand_pose_action)

`list_data` : create data split for training

`generate_tfrecord` : generate tf_record 

`lstm-tfrecord` : '--run_opt' 1 for training, 2 for testing, 3 for evaluation metrics

Using tfrecord is significantly faster than data generation

directories hard-coded in scripts 