# Violence Detection 

## Software Requirements:

* Python 3.6

* Tensorflow 1.11.0 (GPU)

* Cuda 9.0v (update)

* CudaNN 7.3.0 (dependency for Cuda 9.2v)

* sk-video

* scikit-image

* Imgaug

* ffmpeg

* OpenCV (GPU)

* Linux/Unix (Quite good for development then Windows) [Tested on Ubuntu 18.04 and Windows 10 1809 Build]

# Hardware Requirements:

* 8GB 1070 Ti or 1080 Ti (Minimum)
* Above 8GB Ram
* Above 410 version Nvidia Driver
###### Reprocreated work using [this article](http://joshua-p-r-pan.blogspot.com/2018/05/violence-detection-by-cnn-lstm.html) and [git](https://github.com/JoshuaPiinRueyPan/ViolenceDetection)

# Architecture of Project:

   * Train.py: An executable that can train the violence detection models

   * Deploy.py: An executable that can display a video and show if it has violence event per frame.

   * Evaluate.py: An executable that can calculate the accuracies with respect to the given dataset catalog and the model checkpoints.

   * settings/:  A directory that contains various settings in this projects. Most of the commonly changed variables can be found here. I prefer this design philosophy because the user can easily change several variables without get into the source code.

   * src/: Functions and Classes that used by the executables can be found here.

       * src/data: Libraries that deal with data.

       * src/layers: Convenient functions or wrappers for tensorflow. Note: The settings of layers (such as weight decay, layer initialization variables) can be found in settings/LayerSettings.py.

       * src/net: The network blueprints can be found here. We can find examples and design our own networks here. Note: Remember to change the new-developed network by editing the settings/NetSettings.py.

       * src/third_party:  Third-party libraries are placed here. Currently, this folder only contains the data augmentation library

# Training
* Download the fight/non-fight dataset from [here](http://visilab.etsii.uclm.es/personas/oscar/FightDetection/index.html) or, and keep dataset separated in two directories fight/nofights dataset. 

* To make the data catalogs that will tell the data manager where to load the videos, edit the file: tools/Train_Val_Test_spliter.py to specified the path to the dataset videos, the ratio to split the datasets into training, validation and test set. And run such scripts, we will get three data catalogs: train.txt, val.txt, test.txt.

* Edit the settings/DataSettings.py to specify where do we put the data catelogs:
```Shell
	PATH_TO_TRAIN_SET_CATELOG = 'MyPathToDataCatelog/train.txt'
	PATH_TO_VAL_SET_CATELOG = 'MyPathToDataCatelog/val.txt'
	PATH_TO_TEST_SET_CATELOG = 'MyPathToDataCatelog/test.txt'
```

* Edit the settings/TrainSettings.py, and set the variables to fit our environment:
```Shell
	MAX_TRAINING_EPOCH = 30  ##change according to needs of epochs
    EPOCHS_TO_START_SAVE_MODEL = 1  ##starting point of epoch name
    PATH_TO_SAVE_MODEL = "MyPathToSaveTrainingResultsAndModels"

```
*   By default, it will use the G2D19_P2OF_ResHB_1LSTM as its default network. This network is base on the pre-trained Darknet19. The checkpoint of such model is converted from the [Darknet](https://pjreddie.com/darknet/imagenet/) format to the TensorFlow pb format by the use of [Darkflow](https://github.com/thtrieu/darkflow). we can convert the checkpoint  ourself, or download from [here](https://drive.google.com/open?id=1oUPhXtZjTU04DOwAXv6LtQ1GxFG9TD7b).
    * Note: We can change the network by editing the settings/NetSettings.py. Take a look at src/net directory to see the various networks that are available.

* We're ready to train the model:
```Shell
	python3 Train.py
```

# Trained Model 

Pre-trained model for violence detection with video accuracy of 98% and frame accuracy of 97% comprises of hockey+Pecliculas+jogging+walking_datatset_checkpoint click [here](https://drive.google.com/drive/folders/1Tu0sIrcbAOx2vqJMD7tXIj6KkhWuvS5s?usp=sharing) to download. 

# Deploy
After we have trained a model, we can input a video and see its performance by following procedures:
* Edit the settings/DeploySettings.py to set the variables to fit our environment:

```Shell
	PATH_TO_MODEL_CHECKPOINTS = "PathToMyBestModelCheckpoint"
    PATH_FILE_NAME_OF_SOURCE_VIDEO = “PathToTheInputVideo”
    PATH_FILE_NAME_TO_SAVE_RESULT = “NameToTheOutputVideo”

```
* Execute the Violence Detector by the following command:
```Shell
	python3 Deploy.py  ##  command to Deploy over model
```
    * Darknet19 is used for localisation of human in implicit way.
	
# Real Time Detection

* [Line 58 in Deploy.py]: This line convert the image in BGR (if we read the webcam by OpenCV) to the net input. If our OpenCV does not use GPU to do that, it might be kind of slow.

* [Line 79 in src/ViolenceDetector.py]: This line use numpy to stack two images to form one net input. If we find this is the bottleneck maybe we can train a model with single frame image as its input to inference the result. At first, set GROUPED_SIZE=1 in DataSettings.py, then specify to use the G1D19_1Fc_1LSTM model by editing the NetSettings.py.

* [Line 90 in src/ViolenceDetector.py]: This line send the two frames into the network and should be only the process that take most of the time. If this is the case, we can use more powerful graphic cards or, design our own model that is suitable for our device. We can refer to src/net/*.py to see examples to design our own model

# Dataset Creation
We probably need to cut the video clips. This project support cases that:
* The violence event happened in the middle of the video (or, the violence event does not happen from the begin of the video).
* The violence event end before the end of the video.
We can specified the Start Frame and the End Frame of the Violence event of the video in the data catalog (such as the _train.txt_ after we execute the _tools/Train_Val_Test_spliter.py_). In the data catalog, we can find the format shown as follows:
```Shell
	`data/video.avi	<tab> 0.0 <tab> INF`
```
The first element is the path to each video. The second element is the frame index that the Violence event starts. The third element is the frame index that the Violence event ends. For the hockey dataset, the violence start from the begin till the end of the video. Thus the second and third element will be '0' and 'INF', respectively.


