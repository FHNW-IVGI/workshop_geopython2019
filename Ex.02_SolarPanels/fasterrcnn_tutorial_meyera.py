# -*- coding: utf-8 -*-
"""FasterRCNN_Tutorial_MeyerA.ipynb

Original file is located at
    https://colab.research.google.com/ > GitHub > fhnw-ivgi > workshop_geopython2019 > Master > FasterRCNN_Tutorial_MeyerA.ipynb
	https://github.com/FHNW-IVGI/workshop_geopython2019/blob/master/FasterRCNN_Tutorial_MeyerA.ipynb

# Object Detection Tutorial with Faster R-CNN Transfer Learning
## Geopython 2019, Adrian F. Meyer
Parts of this tutorial are based on the [Medium article](https://hackernoon.com/object-detection-in-google-colab-with-custom-dataset-5a7bb2b0e97e) by RomRoc, 2018.

**In this Tutorial we will learn, how to use the Tensorflow Object Detection library, to detect solar panels on tiles of an aerial orthomosaic.**

![alt text](https://wilddrone.ch/wp-content/uploads/2019/06/solardetector.jpg)

The code, libraries and cloud environments used in this tutorial are currently available for free and are generally released open source.
You will need a Google account to execute the Notebook in its entirety, because it is meant to be executed on the Google Colab platform.

*Go to [Google Colab](https://colab.research.google.com/) and upload the notebook there. Make sure that you use Python 3 and GPU hardware acceleration as Runtime Environment.*

The dataset provided is based on the publically available SwissImage orthomosaic by [SwissTopo](https://map.geo.admin.ch/?topic=swisstopo&lang=de&bgLayer=ch.swisstopo.swissimage). 
The images and annotations can be downloaded as Zip File (31 Mbyte) here:
[https://drive.google.com/file/d/1i9RlEJTeB-KRauwhuMp-Vl13-Z3Yrj_y/view?usp=sharing](https://drive.google.com/file/d/1i9RlEJTeB-KRauwhuMp-Vl13-Z3Yrj_y/view?usp=sharing)



![alt text](https://wilddrone.ch/wp-content/uploads/2019/06/Tiles.jpg)

# Download Tensorflow Repo and Python Modules
By executing the first code snippet you initialize your virtual linux-style machine. Use The little arrow ">" in the top left corner to view the file system of your hosted system.
You can use UNIX-style terminal commands by using the prefix % and elevated priviledge commands for installations with the prefix !.
"""

# %cd
  
!git clone https://github.com/tensorflow/models.git
  
"""
This repository contains a number of different models implemented in TensorFlow:
The official models are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.
The research models are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.
The samples folder contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.
"""  

!apt-get install protobuf-compiler python-tk

"""
Protocol buffers are Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data; similar to JSON or XML.
"""

!pip install Cython contextlib2 pillow lxml matplotlib PyDrive
"""
These context modules are necessary python pachages. Especially Cython is important: It allows to call native C or C++ bindings from within python.
"""

!pip install pycocotools
"""
COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. 
"""

# %cd ~/models/research
!protoc object_detection/protos/*.proto --python_out=. #This initializes/compiles the Tensorflow Protobuf evnironment.

import os
os.environ['PYTHONPATH'] += ':/models/research/:/models/research/slim/'

"""# Install Tensorflow on Virtual Machine"""

!python setup.py build
!python setup.py install > /dev/null
"""
This snippet builds and installs the Tensorflow API from the cloned git source.
"""



# %cd slim
!pip install -e .

# %cd ..
!python object_detection/builders/model_builder_test.py
"""
This tests if the installation was successful. The Tests should yield the output [ RUN  OK ]
"""

"""#Upload and Import Dataset"""

# %cd /datalab
!wget http://2019.geopython.net/data/solar.zip
"""
We download the images, annotation files and independent test samples into the datalab folder.
"""

# %cd /datalab
!unzip solar.zip #Scroll through the unzip output to get an idea of the datalab folder content.

"""You can download one of the XML files to check the structure of the PASCAL VOC Annotation format.
Here is an example:



```
<annotation>
	<folder>sol2</folder>
	<filename>solar_10.JPG</filename>
	<path>C:\temp\sol2\solar_10.JPG</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>901</width>
		<height>791</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>solar</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>637</xmin>
			<ymin>152</ymin>
			<xmax>901</xmax>
			<ymax>591</ymax>
		</bndbox>
	</object>
</annotation>

```

We need to write our label name (in this case "solar") into a config file defining all detectable classes.
In our case it is just on class.

Then we iterate through all image files to extract the file names (paths are not relevant) we want to use for training and validation.
"""

# %cd ..
# %cd /datalab

!echo "item { id: 1 name: 'solar'}" > label_map.pbtxt

image_files=os.listdir('images')
im_files=[x.split('.')[0] for x in image_files]
with open('annotations/trainval.txt', 'w') as text_file:
  for row in im_files:
    text_file.write(row + '\n')

"""#Data and Model Preparation
## Generate Bounding Boxes on Images for RPN Network Training
The same process need to be performed with the XML Annotation files.
Additionally, we convert the images to PNG format for easier access.
"""

# %cd /datalab/annotations

!mkdir trimaps

from PIL import Image
image = Image.new('RGB', (901, 791))

for filename in os.listdir('xmls'):
  filename = os.path.splitext(filename)[0]
  image.save('trimaps/' + filename + '.png')

"""##Generate Labelled Tensor Matrices (tf_records)
The Tensorflow Record files contain the actual input data for the Machine Learning process in binary format.
An API specific script can do the job for us.
We use the famous "pet faces model" in our transfer learning process.
We need to split the dataset at this point into training and validation data.
70% of our data (148 of 212 images) will be used for training, the remaining 30% for validation (64 images)
"""

# %cd /datalab

!python ~/models/research/object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=label_map.pbtxt --data_dir=. --output_dir=. --num_shards=1

!mv pet_faces_train.record-00000-of-00001 tf_train.record

!mv pet_faces_val.record-00000-of-00001 tf_val.record

"""##Download the Model Checkpoint you want to use for Transfer Learning
Many different COCO pretrained neural models can be used for bounding box related object detection with Tensorflow.
They all have different advantages or disadvantages (e.g. inferencing speed, accuracy, easy to train, etc.).

An overview can be found with the [TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
"""

# %cd /datalab
!wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  
# %cd /datalab
!tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

# %cd /datalab
!mv faster_rcnn_inception_v2_coco_2018_01_28 pretrained_model

"""##Configure the Paths and Training Parameters
This specifies which files and model checkpoints should be used for the trainings process.
"""

# %cd /datalab

import re

#filename = '/datalab/pretrained_model/pipeline.config'
filename = '/root/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config'
with open(filename) as f:
  s = f.read()
with open(filename, 'w') as f:
  s = re.sub('PATH_TO_BE_CONFIGURED/model.ckpt', '/datalab/pretrained_model/model.ckpt', s)
  s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_train.record-\?\?\?\?\?-of-00010', '/datalab/tf_train.record', s)
  s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_val.record-\?\?\?\?\?-of-00010', '/datalab/tf_val.record', s)
  s = re.sub('PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt', '/datalab/label_map.pbtxt', s)
  f.write(s)

"""# Training on GPU
The execution of this snippet might take a while. Normally the training of 3000 steps should take about 10 minutes (approx. 12 seconds for 100 steps). 

As a rough estimate, the loss value of Faster RCNN models should fall below 0.05 over a few thousand steps and then the training can be aborted. 

We configure automatic termination after 3'000 Steps, in productive trainings as much as 100'000-200'000 Steps can be neccesary.

![alt text](https://cdn-images-1.medium.com/max/800/1*qGb5XNny5G8PrGQ5sejFvw.png)

This graph shows the expected training progess of the model with the supervision tool "Tensorboard".
"""

# %cd /datalab

# %cp -R /datalab /content

!python ~/models/research/object_detection/model_main.py \
    --pipeline_config_path=/root/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config \
    --model_dir=/datalab/trained \
    --train_dir=/datalab/trained \
    --logtostderr \
    --logdir=/datalab/trained \
    --num_train_steps=3000 \
    --num_eval_steps=500 \
    --max_evals=0

"""# Export Inference Graph
Inferencing means to apply the model to images which haven't been used for training.

We reserved a few images to check if our model performs correctly.

The frozen Inference Graph gets generated from the last model checkpoint and contains all elements of the model neccesary to perform inference (also on weaker hardware), but it cannot be used to continue training the model.
"""

# %cd /datalab

lst = os.listdir('trained')
lf = filter(lambda k: 'model.ckpt-' in k, lst)
last_model = sorted(lf)[-1].replace('.meta', '')

!python ~/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=/root/models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config \
    --output_directory=fine_tuned_model \
    --trained_checkpoint_prefix=trained/$last_model

"""##Alternative: Import a Finished Inference Graph
Uncomment this section, if you want to download a finished inference graph if you could not finish the training.
"""

#Alternative to training, you can download a fully trained model inference graph by uncommenting (#) the following lines

#%cd /datalab
#!wget http://2019.geopython.net/data/trained.tar.gz
#%cd /datalab
#!tar -xzf trained.tar.gz

"""# Run Inference"""

# %cd /root/models/research/object_detection




import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#if tf.__version__ < '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  

  
  
# This is needed to display the images.
# %matplotlib inline




from utils import label_map_util

from utils import visualization_utils as vis_util




# What model to download.
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/datalab/fine_tuned_model' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/content/datalab', 'label_map.pbtxt')

NUM_CLASSES = 37




detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
    
    
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/datalab/testsamples/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.JPG'.format(i)) for i in range(1, 4) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict




for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)

"""## Run Inference on Additional Samples"""

# %cd /root/models/research/object_detection




import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#if tf.__version__ < '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  

  
  
# This is needed to display the images.
# %matplotlib inline




from utils import label_map_util

from utils import visualization_utils as vis_util




# What model to download.
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/datalab/fine_tuned_model' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/content/datalab', 'label_map.pbtxt')

NUM_CLASSES = 37




detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
    
    
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/datalab/images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'solar_{}.JPG'.format(i)) for i in range(150, 166) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict




for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)

