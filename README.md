![](./yolov11.webp)

# Traffic Sign Detection using YOLOv11

- [Traffic Sign Detection using YOLOv11](#traffic-sign-detection-using-yolov11)
  - [Data](#data)
  - [Model](#model)
  - [Fine Tuning](#fine-tuning)
    - [YOLO11n summary (fused)](#yolo11n-summary-fused)
    - [Run summary](#run-summary)
  - [Detections](#detections)
  - [Dependencies](#dependencies)
  - [Project Setup](#project-setup)
  - [Limitations](#limitations)
  - [Conclusion](#conclusion)
  - [Acknowledgements](#acknowledgements)

## Data

The [Self-Driving Cars Dataset](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou/dataset/6) is used to train the traffic sign detection model. It contains **4969** total images
split into train, val and test sets with **3530**, **801** and **638** images of dimension `416x416` respectively. The dataset contains images of 15 different traffic signs.

The classes available in the dataset are:

42 signs according to the GTSRB dataset used for training

## Model

The `yolo11n` version of the model is used to fine-tune on the dataset. The model was trained for **50** epochs with batch size **16**.

*Note*: The .ipy notebook is not uploaded due to privacy issues.



5. Road Network Analysis: Transportation engineering researchers can use this model to analyze how efficiently different sign classes are distributed and recognized around the city. This data can be instrumental in planning more efficient and safer road networks.

## Acknowledgements

The media files used to test the model predictions are taken from [pexels.com](https://www.pexels.com/).
