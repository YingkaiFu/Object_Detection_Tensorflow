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
#### Dataset prepare

To train your our object detection model, 

### Domo to lauch
Run main.py in this project