# CarND-TrafficLight-Detection

## Step 1 - Get the Data

1. Gather the data - We have 2 sets of data - simulator and real-world data. 
2. Label and annotate the data

We will come to download the dataset in step 4 - create the TFRecord.

## Step 2 - Setup the tensorflow models

1. Do `git clone https://github.com/tensorflow/models.git` inside the tensorflow directory.

2. We will work with python 2, so activate respective virtual environment.

3. Follow the instructions at [this page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for installing some simple dependencies. See rest of points as summary of commands.

4. Go to research directory - `cd tensorflow/models/research/` and run following commands:

   ```pyt
   python setup.py build
   python setup.py install
   python slim/setup.py build
   python slim/setup.py install
   protoc object_detection/protos/*.proto --python_out=.
   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
   ```

5. Now that installation is done, test it:

   ```pyt
   python object_detection/builders/model_builder_test.py
   ```


## Step 3 - Download the Model

Here is the list of [pre-trained models zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Download the required model tar.gz files and untar them into `/tensorflow/models/research/` directory with `tar -xvzf name_of_tar_file`.

Next we need to setup an [object detection pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). TensorFlow team also provides [sample config files](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) on their repo. For the training,we can use any of three models, `ssd_mobilenet_v1_coco`,  *ssd_inception_v2_coco* and *faster_rcnn_resnet101_coco*. These models can be downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Here are the download link:

```Sh
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

**Select the model**

We can go ahead with the `ssd_inception_v2_coco` model.

## Step 4 - Create TFRecord files

`python data_conversion_udacity_sim.py --output_path sim_data.record`

`python data_conversion_udacity_real.py --output_path real_data.record`

Step 1 and Step 4 has been taken care off: https://github.com/kinshuk4/CarND-TrafficLight-Detection-Dataset. Just clone/download the repo, and use the data folder in it. 

## Step 5 - Train the Model

Commands for training the models and saving the weights for inference. 

### Use the label map

We need to use labelmap configured in `label_map.pbtxt` which contains 4 classes:

```Json
item {
  id: 1
  name: 'Green'
}

item {
  id: 2
  name: 'Red'
}

item {
  id: 3
  name: 'Yellow'
}

item {
  id: 4
  name: 'off'
}
```

### Use the config files

TensorFlow team also provides [sample config files](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) on their repo. We have to setup the config files for real and simulator data.

### Training Steps

Now for all the models, we do following steps:

1. [COCO pre-trained network](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) models
2. The TFRecord files we created earlier
3. The label_map file with our classes
4. The image data-set
5. The [TensorFlow model API](https://github.com/tensorflow/models)

### Use Inception SSD v2

#### For Simulator Data - Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### For Simulator Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_models/frozen_sim_inception/`

#### For Real Data - Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --train_dir=data/real_training_data`

#### For Real Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_inception/`

### Use Faster-RCNN model

#### For Simulator Data - Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### For Simulator Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_sim/`

#### For Real Data - Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --train_dir=data/real_training_data`

#### For Real Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_real/`

### Use MobileNet SSD v1

(Due to some unknown reasons the model gets trained but does not save for inference. Ignoring this for now.)

#### For Simulator Data - Training

`python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### For Simulator Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_models/frozen_sim_mobile/`

#### For Real Data - Training

`python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet-traffic_udacity_real.config --train_dir=data/real_training_data`

#### For Real Data - Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_mobile/`



### Test the Model

Working on it.

**Credits**

To save the time and effort, I followed the post from https://becominghuman.ai/@Vatsal410 and https://medium.com/@anthony_sarkis, where they have shared shared his annotated data-set openly available for all to use. Also, I got the config files for the project as well, that saved lot of time. Thanks for sharing such a nice post and the dataset.
