# Realization detection of microfluidics droplets on video from microscope


# Quick Start
For get quick results you can use our datasets, weights or samples. So, in the beginning, you need download yolov7 implementation[![Yolo](https://img.shields.io/badge/WongKinYiu-yolov7-brightgreen)](https://github.com/WongKinYiu/yolov7).

## Training
If you want to train model yourself, use our processed data by link: 
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
If you want to detect droplets whith your or our weights, you can use our video samples by link
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

If you want to add an image for detection, we recommend you using standart yolo script _detect.py_, becouse our method needs a video for 
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
- Select **Floor**, **Materials Option** tab, **Floor** Material and select needed image in browse window
- Запустите Generator.blend, предварительно установив актуальную версию **Blender**
[![Blender Download](https://img.shields.io/badge/Blender-3.1.2-brightgreen)](https://www.blender.org/download/)

- Вкладка _Scripts_ -> скрипт _CreateScene_
- Установите удобные вам параметры в скрипте согласно комментариям (По умолчанию генерируются 90-160 объектов и
 производится 1 рендер)
- **Обращаем ваше внимание, что генерация происходит в главном потоке по умолчанию в режиме CPU, при рендере
 вы не сможете пользоваться Blender**
 <details>
  <summary>Новый проект Blender</summary>
 
  Если вы хотите запустить скрипты в другом проекте:
  - Предварительно выполните скрипт _SceneOption_
  - Настройте альфа-каналы созданных материалов через вкладку _Shading_
  - При необходимости добавте интересующий вас фон на камеру и в настройки рендера
</details>


# Разметка данных
Вы можете использовать наши размеченные данные, их будет достаточно для пробного обучения моделей
Так же вы можете использовать любое ПО для создания _Anchor Boxes_, например 
[![LabelIMG](https://img.shields.io/badge/tzutalin-labelIMG-brightgreen)](https://github.com/tzutalin/labelImg).
Так же вы можете воспользоваться авторазметкой от авторов (скрипт _AnchorOcto_, по умолчанию создает файлы меток для Yolo).
Если вы планируете размечать данные вручную, советуем вам использовать наш проект .blend и создавать октаэдрические формы
в красном цвете. Тогда вы сможете отметить на рендерах объекты даже с учетом перекрытия, а далее перевести всю тренировочную
выборку в ч/б формат.


# Использование модели
Для тестирующих обнаружений вы можете использовать наши веса для Yolo v5.
1. Скачайте актуальный репозиторий Yolo v5
[![Yolo](https://img.shields.io/badge/WongKinYiu-yolov7-brightgreen)](https://github.com/WongKinYiu/yolov7)
2. Добавьте наши веса last.pt в _yolov5->runs->train->exp->weights_ (если таких нет, то создайте) и используйте 
   этот код (Для collab/notebook), либо в строке кода измените путь к весам
```python
                               #your path to weights                                                 path to img
!python detect.py --weights .../yolov5/runs/train/exp/weights/last.pt --img 224 --conf 0.25 --source ../58.png 
```
3. Следуйте по указанному пути к обработанному изображению (обычно это _yolov5->runs->detect->exp{num your exp}->yourfile.png_)


