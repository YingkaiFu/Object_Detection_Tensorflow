### Object detection for 20 items with tensorflow
This project is based on tensorflow object detection API aimded
 to detect objects which place on the desk. The training items
 can be found in the file `item_list.pdf`, they can overrlapped
 with each others. Our model finetunes from faster-r-cnn 
 resnet-51, the training is done with two Tesla V100 GPU on
 Tensorflow 1.8 
 
![Demo](https://github.com/YingkaiFu/Object_Detection_Tensorflow/blob/master/detect_result.jpg)
### Domo to lauch
Download the model from [here](https://pan.baidu.com/s/1H6KovubBhVQMqz6P9FqVow)
, move the model to the root folder of the project and then
run main.py

### Packages
* Tensorflow-GPU
* opencv-python
* Python3.6
* PyQt5
### How to train with your own data
#### Make the label
Modify the file `label.pbtxt` to suit your data. for example, if you
have three items to recognize:item1,item2,item3, please edit the file
in this format.
```
item{
  id: 1
  name: 'item1'
}
item{
  id: 2
  name: 'item2'
}
item{
  id: 3
  name: 'item3'
}
```
#### Dataset prepare
To train your our object detection model, we have to make our 
dataset, object detection needs pictures as input and the item
id, probability, box and the output, so we have to create our
model with those information. [lableImg](https://github.com/tzutalin/labelImg.git)
can be used to create the required dataset, it outputs a series of `XML` 
files.
#### Transform xml to csv
In this step, we have to transform the xml file into csv file for
later process. we can use the script `xml_csv.py` to finish this step.
#### Transform csv to record
Now we have to transform csv files to record format which can be recognized
by tensorflow, modifiy the function `class_text_to_int` and the
data path in script `generate_TF.py` to fit your own data. make sure 
the content in `class_text_to_int` can be mapped with the file `label.pbtxt`'
### Choose the pre-trained model
Tensorflow Model Zoo provide us with many good [models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md),
you can choose your model according to the request of your project. For
instance, Faster-r-cnn have higher precision but it cost time to train and
inference, it also consumes large GPU memory, in the contrast, `SSD`
is easy to train and the model can also be deployed on mobile phones.
In my project, I use Faster-r-cnn resnet51 as the pre-trained model.
### Modify configurations
After choose a model, we have to add a file to let tensorflow know how
to train and test the model. We can choose and modify the
configuration file in object_detection/samples/configs. You should
modify the class number and relevant train and eval path together
with the fine_tune_checkpoint.
```
    num_classes: 90
    
    ...
    fine_tune_checkpoint: "/home/yingkai/faster_rcnn_nas_coco_2018_01_28/model.ckpt"
    
    ...
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100"
    label_map_path: "/home/yingkai/Project/PycharmProjects/Tensorflow/object_detection/training/label.pbtxt"
    
```
### Train your model
Now everything is ready, run the following command to train the model,
make sure the path is correct.
```angular2html
python object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
where `train_dir` is the output dir, `pipeline_config_path` is your configuration path.
### Export the model
To use the model without the configuration, we can export the model
so it can run everywhere where tensorflow is installed, try the following
command.
```angular2html
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
After some time, a file named `frozen_inference_graph.pb` will appear in your
output directory.
### Run your model!
Modify the path in detect.py to your own path, use your own model and
then try the object detection by running main.py
