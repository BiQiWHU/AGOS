# All Grains, One Scheme (AGOS)
Offcial implementation of our AGOS framework for remote sensing scene classification.

This work has been published in ```IEEE Transactions on Geoscience and Remote Sensing``` entitled as ```All Grains, One Scheme (AGOS): Learning Multi-grain Instance Representation for Aerial Scene Classification```.

# Project Overview

The key idea of this prject is to extend the deep multiple instance learning into a multi-grain form. The multi-grain multi-instance learning is suitable to describe the varied object sizes and scale differences in remote sensing scenes. An technique framework of the proposed method is attached as follows.

![avatar](/framework.png)

# Enviroment dependency

The code is implemented on top of the Python3, and there are only a few dependencies that a development need to config.
Before starting the code, please ensure the below packages and the corresponding versions are available.
```
Python > 3.5

Tensorflow > 1.6

OpenCV > 3

Numpy > 1.16
```
The datasets this paper use are all publicly available, and can be found in 
<a href="https://captain-whu.github.io/AID/"> AID</a>,
<a href="http://weegee.vision.ucmerced.edu/datasets/landuse.html"> UCM</a>, and 
<a href="http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html"> NWPU</a>, respectively.

The ResNet-50 pre-trained model can be downloaded from <a href="https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"> here</a> and is supposed to put into the ```checkpoint``` file folder.

# How to run the code?
For training:

Step 1, run the ```tfdata.py``` file to transfer the data into the tf.record file format.
```
python tfdata.py
```

Step 2, run the ```training.py``` file to start training.
```
python training.py
```

For testing:

Run the ```test1.py file``` to test the performance of a single model.
```
python test1.py
```

Run the ```testall.py``` file to test the performance of all the models in the checkpoints file folder. 
```
python testall.py
```

Please note, before using the ```testall.py``` script, please remember to delete a file named ```checkpoint``` in the ```checkpoints``` file folder.

# Citation and Reference
If you find this project useful, please cite:
```
@ARTICLE{Bi2022AGOS,
  author={Bi, Qi and Zhou, Beichen and Qin, Kun and Ye, Qinghao and Xia, Gui-Song},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={All Grains, One Scheme (AGOS): Learning Multigrain Instance Representation for Aerial Scene Classification}, 
  year={2022},
  volume={60},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2022.3201755}}
```

  Other our former works related to Deep MIL may also be cited as:
  ```
  @ARTICLE{Bi2021MIDCNet,
  author={Bi, Qi and Qin, Kun and Li, Zhili and Zhang, Han and Xu, Kai and Xia, Gui-Song},
  journal={IEEE Transactions on Image Processing}, 
  title={A Multiple-Instance Densely-Connected ConvNet for Aerial Scene Classification}, 
  year={2020},
  volume={29},
  pages={4911-4926},
  doi={10.1109/TIP.2020.2975718}}
  ```
  
# Contact Information

Qi Bi

q_bi@whu.edu.cn   2009biqi@163.com
