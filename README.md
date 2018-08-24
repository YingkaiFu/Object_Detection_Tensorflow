### Object detection for 20 items with tensorflow
This project is based on tensorflow object detection API aimded
 to detect objects which place on the desk. The training items
 can be found in the file 'item_list.pdf', they can overrlapped
 with each others. Our model fine tunes from faster-r-cnn 
  resnet-51, the training is done with two Tesla V100 GPU on
  Tensorflow 1.8 
    
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

### Domo to lauch
Run main.py in this project