# Privacy-Preserving Solution for Securing Gatherings with Computer Vision

## About
This was my master graduation project at Frankfurt School under supervision of Levente Szabados and Florian Ellsaesser.

The goal of this study is to build a face recognition system that can be used to secure gatherings in a privacy-
preserving manner. The system aims to classify entrants' faces captured in a camera into a _registered_ or 
an _outsider_ based on anonymized face embeddings and in compliance with GDPR. 

## Pipeline
The pipeline of our face recognition system consists of two steps: (1) face detection; and (2) face recognition. 
In the first step, two different pre-trained face detectors were used which are HOG and RBF. The analysis shows that 
the performance of RBF outperforms HOG in both accuracy and computation time. The second step was further divided into 
two steps which are face embeddings and face similarity. Different quality face embeddings were generated by 
pre-trained face-recognition models such as ResNet and FaceNet. Analysis confirms that calculating face similarity 
based on high-quality face embeddings obtained from ResNet provides higher accuracy compared to low-quality face 
embeddings provided by FaceNet. 

## Challenges
Using low-quality face embeddings was preferred to reduce the computational time required for real-time analysis. 
However, building a face similarity model that can adapt to these low-quality face embeddings was a challenging task. 
Average N-Way test accuracy for N-Way ranges from 1 to 50 shows that relying on L2-distance alone could only provide 
35% accuracy for low-quality face embeddings. On the other hand, using L2-distance for face verification on high-
quality face embeddings provided an average N-Way test accuracy of 90% in the same N-Way range. However, high-quality 
face embeddings were not preferred due to their higher time complexity.

## Contribution
A novel approach was suggested to perform additional learning beside L2-distance to adapt to low-quality face 
embeddings. Four different Siamese architectures based on neural networks were designed to take pairs of face 
embeddings to perform binary classification. It was found out that, for low-quality face embeddings, adding dense 
layers after L2-distance layers increased average N-Way test accuracy from 35% to 96% compared to only calculating 
L2-distance.

## Implementation
The face recognition system is implemented in Python using PyCharm as an Integrated Development Environment (IDE). 
In addition to standard Python libraries, different open-source libraries were used such as Numpy, SQLite3, OpenCV, 
PIL, Dlib, Tensorflow, and ONNX. In this section, the most important components of the face recognition system are 
explained and supplemented with a few examples. The face recognition system functionality can be accessed via a 
command-line API which is shown in [FaceDetect.py](FaceDetect.py) and [FaceDataBase.py](FaceDataBase.py).

The command-line API allows using the system to detect faces in images or videos and to extract face embeddings which 
then can be saved in the system face database (FDB). In addition, the system allows to detect faces in real-time and 
check if these faces are registered in FDB or not based on face similarity Siamese models.

### Face Detection
Face detection is a regression process to detect one or more human face locations in images or videos. Face locations 
are expressed as 4 points in images' dimension (i.e. pixels) which defines faces boundaries (i.e. left, right, top, 
bottom). Face detectors provide a list of face locations based on the number of detected faces. 

Two pre-trained face detectors were implemented in our face recognition system which are Histograms of Oriented 
Gradients (HOG) and Receptive Field Block Net (RBF). HOG model is implemented in [dlib](http://dlib.net/) library as 
[`get_frontal_face_detector()`](http://dlib.net/python/index.html\#dlib.get_frontal_face_detector). The same function 
is used in [face-recognition](https://face-recognition.readthedocs.io/en/latest/index.html) library under 
[`face_locations()`](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations) 
function. The implementation of RBF model for face detection can be found under the github repository named 
[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). 
HOG is a classical method while RBF is a modern deep-learning-based approach.

To try these models on an image file using our system, the following command line is entered in the shell:
```python
>>> python FaceDetect.py image --path=input/imgs/crowd.jpg --trans=1 --detector=hog --save
```
This will run the Python program named \verb|FaceDetect.py| with \verb|image| as an argument with the following options: 
- `--path`: to enter the image path.
- `--save`: to save the results.
- `--trans`: to specify face mask transparency.
- `--detector`: to specify the face detector `hog` or `rbf`.

The Python program [FaceDetect.py](FaceDetect.py) with argument `image` calls 
[face_detect.py](utilities/face_detect.py) and uses `detect_faces_in_images()` function. This function opens 
the image file and then run the face detector to find face locations. Consequently, bounding boxes around faces are 
drawn as face masks using another function called `face_locations()` under 
[face_draw.py](utilities/face_draw.py) . 

## References
1. [Face Recognition](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
1. [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
1. [Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
1. [OpenCV Face Recognition](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
