from utilities import face_utilities, detect_imgs_onnx
from face_recognition import load_image_file
import os
import time

if __name__ == '__main__':
    # Load the jpg file into a numpy array
    # image_path = "input/people.jpg"
    # face_utilities.find_face_locations(image_path, face_detector='ultra-light', report=True)
    # face_utilities.find_face_landmarks(image_path, enumerate_faces=True, report=True)
    # detect_imgs_onnx.find_face_locations(image_path, enumerate_faces=True, report=True)

    label_path = "models/voc-model-labels.txt"
    result_path = "./detect_imgs_results_onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]
    path = "imgs"
    sum = 0
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(path)
    for file_path in listdir:
        image_path = os.path.join(path, file_path)
        time_time = time.time()
        face_utilities.find_face_locations(image_path, face_detector='hog', report=True)
        print("cost time:{}".format(time.time() - time_time))


