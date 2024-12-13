# Realization detection of microfluidics droplets on video from microscope


# Quick Start
For get quick results you can use our datasets, weights or samples. So, in the beginning, you need download yolov7 implementation[![Yolo](https://img.shields.io/badge/WongKinYiu-yolov7-brightgreen)](https://github.com/WongKinYiu/yolov7).

## Training
If you want to train model yourself, use our processed data by link:  <a href="https://drive.google.com/drive/folders/104Orc2h7frr11w__hI5zGlNCHhpYgnnG?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>

1. Then you need download **detection** and **train** folders in this repo. 
2. After that, add all files from detection folder to root yolov7. The file _custom_plots_ needs 
    to be added along the path _yolov7_ -> _utils_.
3. Open the file _blob.yaml_ from **train** folder and change paths to sets how as in the example in the file
4. Translate _blob.yaml_ in _yolov7_ -> _data_ folder
5. Now you can start training model. Use this line  in jupyter notebook as an example (Windows):
 ```python
!python train.py --workers 8 --epochs 1 --device 0 --batch-size 16 --data data/blob.yaml --img 640 360 --cfg cfg/training/yolov7.yaml --weights '' --name blob_test --hyp data/hyp.scratch.custom.yaml
```
## Detection
If you want to detect droplets whith your or our weights, you can use our video samples by link: <a href="https://drive.google.com/drive/folders/16pVv_uIjr1gZ4TzCAb64D3J0CZCiEXLo?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
1. Then you need download **detection** folder in this repo.
2. After that, add all files from detection folder to root yolov7. The file _custom_plots_ needs 
    to be added along the path _yolov7_ -> _utils_.
1. Just use this line:
 ```python
                           #path to weights                                  #path to video
!python custom_detect.py --weights best.pt --conf 0.2 --img-size 640 --source BlobV.mp4
```

**Example**

![Detect](https://media.giphy.com/media/ozVWYbr73cRaDuMjWY/giphy.gif)

If you want to add an image for detection, we recommend you using standart yolo script _detect.py_, because our method needs a video for 
calculate dynamic characteristics.

# Step by Step
Also you can repeat our research step by step.
## Generation
At first, you need to generate synthetic data for further model training.
1. Download **generation** folder from this repo.
2. Download and install Blender [![Blender Download](https://img.shields.io/badge/Blender-3.1.2-brightgreen)](https://www.blender.org/download/)
3. Open the _blob2.blend_ file and click on **Scripting** tab.
4. Select range for generating images count
5. Write path to folders, where yo want to save images

**Example**


<a href="https://freeimage.host/ru"><img src="https://iili.io/HqoaYss.png" alt="HqoaYss.png" border="0"></a>


6. Start generating

We recommend you to do test on 1 picture. If you get generated images without substrate, then add it manually. 
- Select **Floor**, **Materials Option** tab, **Floor** Material and select needed image in browse window.
Also, if you do not have a powerful PC, then generate images in small batches (100-200 images at a time).
## Marking
Second step is labeling the created images in yolo format
1. Download the marking folder from this repo.
2. Open the marking project in any IDE (we recommended PyCharm)
3. Open the _JpegARc.py_ script and add the path to the folder with your color images and the path to the folder where you want to save the noisy images.

**Example**


[![HCh5t5P.md.png](https://iili.io/HCh5t5P.md.png)](https://freeimage.host/i/HCh5t5P)


4. Run script.
5. Open the _main.py_ script and add the path to the folder with your masks and the path to the folder where you want to save the txt files with labels.


**Example**


[![HChG87f.md.png](https://iili.io/HChG87f.md.png)](https://freeimage.host/i/HChG87f)


6. Run script.

## Training
1. In the beginning, you need download yolov7 implementation[![Yolo](https://img.shields.io/badge/WongKinYiu-yolov7-brightgreen)](https://github.com/WongKinYiu/yolov7).
2. Download **train** folder in this repo. 
3. You need to prepare the folder hierarchy in the generated dataset. For example, see our data by link: <a href="https://drive.google.com/drive/folders/104Orc2h7frr11w__hI5zGlNCHhpYgnnG?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
4. Open the file _blob.yaml_ from **train** folder and change paths to your sets how as in the example in the file
4. Move _blob.yaml_ in _yolov7_ -> _data_ folder
5. Now you can start training model. Use this line  in jupyter notebook as an example (Windows):
 ```python
        #you can change hyperparams to increase quality of your result
!python train.py --workers 8 --epochs 1 --device 0 --batch-size 16 --data data/blob.yaml --img 640 360 --cfg cfg/training/yolov7.yaml --weights '' --name blob_test --hyp data/hyp.scratch.custom.yaml
```
## Coefficient calculation
1. Download **coefficient** folder from this repo.
2. Open the coefficient project in any IDE (we recommended PyCharm)
3. Open the _channel_size.py_ script.
4. Add in the project folder any frame from your sample for detection (you can use our samples by link: <a href="https://drive.google.com/drive/folders/1c4oT04RmMVdZzgnwagMsZKmGaJl4jRB2?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>).


**Sample**


[![HChp04p.md.jpg](https://iili.io/HChp04p.md.jpg)](https://freeimage.host/i/HChp04p)


5. Add the original channel size in _coeff_ variable (in our example 500 μm, you can leave unchanched this parameter if you just repeat our research)
6. Change the _target_height_ variable. This is the video height resolution for detection.

**Example**

[![HCjxnOG.md.png](https://iili.io/HCjxnOG.md.png)](https://freeimage.host/i/HCjxnOG)



7. Run script and move created _coef.txt_ to root yolov7 folder.

## Detection

1. Download **detection** folder in this repo.
2. After that, add all files from detection folder to root yolov7. The file _custom_plots_ needs 
    to be added along the path _yolov7_ -> _utils_.
3. Just use this line:
 ```python
                           #path your to weights                             #path to video sample
!python custom_detect.py --weights best.pt --conf 0.2 --img-size 640 --source BlobV.mp4
```

**4. Get result!**



![NDetect](https://media.giphy.com/media/a2KgmdaYwCODPhjd2u/giphy.gif)
![DetectB](https://media.giphy.com/media/Zx2pj9GPX76Smn1fSL/giphy.gif)

# Links
- Frames from video <a href="https://drive.google.com/drive/folders/1c4oT04RmMVdZzgnwagMsZKmGaJl4jRB2?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
- Video samples <a href="https://drive.google.com/drive/folders/16pVv_uIjr1gZ4TzCAb64D3J0CZCiEXLo?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
- Dataset <a href="https://drive.google.com/drive/folders/104Orc2h7frr11w__hI5zGlNCHhpYgnnG?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
- Generated data <a href="https://drive.google.com/drive/folders/1fiL_nwUzRA6ryYSrHCgxOVQVZJdGT8Ny?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>
- Detection examples <a href="https://drive.google.com/drive/folders/1v1G7n_1RW7yx6OzoBfC3ip4Bxci6BDjI?usp=sharing"><img src="https://iili.io/HCNp5Wg.th.png" width="32" height="32"></a>

# Citations
- Kartashov, O.O.; Chapek, S.V.; Polyanichenko, D.S.; Belyavsky, G.I.; Alexandrov, A.A.; Butakova, M.A.; Soldatov, A.V. Online Microfluidic Droplets Characterization Using Microscope Data Intelligent Analysis. Big Data Cogn. Comput. 2023, 7, 7. https://doi.org/10.3390/bdcc7010007
# Acknowledgements
- https://github.com/WongKinYiu/yolov7
