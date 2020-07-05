import face_recognition
import os
import re
import time
import PIL
import cv2
import numpy as np
import onnx
from utilities import face_database
from utilities import face_draw
import utilities.vision.utils.box_utils_numpy as box_utils
# from caffe2.python.onnx import backend
# onnx runtime
import onnxruntime as ort


def onnx_order_face_locations(face_locations):
    # draw function needs this order: [right, bottom, left, top]
    return [[right, bottom, left, top] for [top, right, bottom, left] in face_locations]


def onnx_predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def onnx_image_preprocessing(image_file, size=(640, 480)): #(320, 240)

    if type(image_file) == str:
        image = cv2.imread(image_file)
    else:
        image = image_file

    # Add if there is a problem with the model onnx_predict
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_image = cv2.resize(image, (size[0], size[1]))
    image_mean = np.array([127, 127, 127])
    processed_image = (processed_image - image_mean) / 128
    processed_image = np.transpose(processed_image, [2, 0, 1])
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image.astype(np.float32)

    if type(image_file) == str:
        return processed_image, image
    else:
        return processed_image


def detect_faces(images_folder, model='onnx', lib='pil', report=True, show_images=True, save_images=True,
                 label_faces=True):

    # model preparation
    if model == 'onnx':
        label_path = "utilities/models/voc-model-labels.txt"
        class_names = [name.strip() for name in open(label_path).readlines()]

        onnx_path = "utilities/models/onnx/fixed_version-RFB-640.onnx"
        predictor = onnx.load(onnx_path)
        # onnx.checker.check_model(predictor)
        # onnx.helper.printable_graph(predictor.graph)
        # predictor = backend.prepare(predictor, device="CPU")  # default CPU
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        threshold = 0.7

    total_processing_time = 0

    if not os.path.isdir(images_folder):
        image_file = os.path.basename(images_folder)
        images_folder = os.path.dirname(images_folder)
        listdir = [image_file]
        return_flag = True
    else:
        listdir = os.listdir(images_folder)
        return_flag = False

    images_folder_result = "output" + re.split('input',images_folder)[1] + "_result"
    if not os.path.exists(images_folder_result):
        os.makedirs(images_folder_result)

    if report:
        print('-' * 60)
        print("{:<20s}{:>20s}{:>20s}".format('image-file', 'num-of-faces', 'process-time(sec)'))
        print('-' * 60)

    for image_file in listdir:

        start_time = time.time()

        image_path = os.path.join(images_folder, image_file)
        result_image_path = os.path.join(images_folder_result, image_file)

        # todo: order should adapt to onnx
        if model == 'onnx':
            processed_image, image = onnx_image_preprocessing(image_path)
            confidences, boxes = ort_session.run(None, {input_name: processed_image})
            face_locations, labels, probs = onnx_predict(image.shape[1], image.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            labels_probs = [(f"{class_names[labels[i]]}({probs[i]:.2f})") for i in range(len(face_locations))]
        elif model == 'hog':
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model='hog')
            labels_probs = [f"face(?)" for i in range(len(face_locations))]

        number_of_faces = len(face_locations)
        processing_time = round((time.time() - start_time), 2)
        total_processing_time += processing_time

        if report:
            print("{:<20s}{:>20d}{:>20.2f}".format(image_file, number_of_faces, processing_time))

        report_values = (result_image_path, processing_time, labels_probs)
        face_draw.draw_face_locations(image=image, face_locations=face_locations, report_values=report_values, lib=lib,
                                      show_images=show_images, save_images=save_images, return_images=False,
                                      label_faces=label_faces, show_axes=False, show_points=False)

        '''
        save is part of draw_face_locations function
        if save_images:
            if lib == 'pil':
                if type(labeled_image) == np.ndarray:
                    labeled_image = PIL.Image.fromarray(labeled_image)
                labeled_image.save(result_image_path)
            elif lib == 'cv2':
                cv2.imwrite(result_image_path, labeled_image)
        '''
    if report:
        print("Total processing time (seconds): ", round(total_processing_time, 2))

    if return_flag:
        return face_locations


def landmark_faces(images_folder, lib='pil', report=True, show_images=True, save_images=True):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param label_faces: Show number of the face according to detection order
    :param face_landmarker: - "large" (default) or "small" which only returns 5 points but is faster
    :return:
    """

    total_processing_time = 0

    if not os.path.isdir(images_folder):
        image_file = os.path.basename(images_folder)
        images_folder = os.path.dirname(images_folder)
        listdir = [image_file]
        return_flag = True
    else:
        listdir = os.listdir(images_folder)
        return_flag = False

    images_folder_result = "output" + re.split('input',images_folder)[1] + "_result"
    if not os.path.exists(images_folder_result):
        os.makedirs(images_folder_result)

    if report:
        print('-' * 60)
        print("{:<20s}{:>20s}{:>20s}".format('image-file', 'num-of-faces', 'process-time(sec)'))
        print('-' * 60)

    for image_file in listdir:

        start_time = time.time()

        image_path = os.path.join(images_folder, image_file)
        result_image_path = os.path.join(images_folder_result, image_file)

        image = face_recognition.load_image_file(image_path)
        face_landmarks = face_recognition.face_landmarks(image, model="large")

        number_of_faces = len(face_landmarks)
        processing_time = round((time.time() - start_time), 2)
        total_processing_time += processing_time

        if report:
            print("{:<20s}{:>20d}{:>20.2f}".format(image_file, number_of_faces, processing_time))

        report_values = (result_image_path, processing_time)
        face_draw.draw_face_landmarks(image, face_landmarks, report_values, lib, show_images, save_images, label_faces=True)

    if report:
        print("Total processing time (seconds): ", round(total_processing_time, 2))

    if return_flag:
        return face_landmarks


def find_face_embeddings(images_folder, report=True, show_images=False, return_dict=False):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param face_detector:   - "hog" is less accurate but faster on CPUs.
                            - "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available).
                            - The default is "hog".
    :return: face_locations
    """

    total_processing_time = 0

    if not os.path.isdir(images_folder):
        image_file = os.path.basename(images_folder)
        images_folder = os.path.dirname(images_folder)
        listdir = [image_file]
    else:
        listdir = os.listdir(images_folder)

    images_folder_result = "output" + re.split('input',images_folder)[1] + "_result"

    if not os.path.exists(images_folder_result):
        os.makedirs(images_folder_result)

    if report:
        print('-' * 80)
        print("{:<20s}{:>20s}{:>20s}{:>20s}".format('image-file', 'num-of-faces', 'num-of-embeddings',
                                                    'process-time(sec)'))
        print('-' * 80)

    if return_dict:
        face_utilities_dict = {'face': [], 'path': [], 'location': [], 'embedding': []}

    for image_file in listdir:

        start_time = time.time()

        image_path = os.path.join(images_folder, image_file)
        result_image_path = os.path.join(images_folder_result, image_file)

        image = face_recognition.load_image_file(image_path)
        face_locations = detect_faces(image_path, report=False, show_images=show_images, save_images=False)
        face_embeddings = face_recognition.face_encodings(image, known_face_locations=face_locations, model='small')

        processing_time = round((time.time() - start_time), 2)
        total_processing_time += processing_time

        if report:
            print("{:<20s}{:>20d}{:>20d}{:>20.2f}".format(image_file,
                                                          len(face_locations),
                                                          len(face_embeddings),
                                                          processing_time))

        if return_dict:
            for i in range(len(face_embeddings)):
                face_utilities_dict['face'].append(image_file + os.path.sep + 'face_' + str(i))
                face_utilities_dict['path'].append(image_path)
                face_utilities_dict['location'].append(face_locations[i])
                face_utilities_dict['embedding'].append(face_embeddings[i])

    if report:
        print("Total processing time (seconds): ", round(total_processing_time, 2))

    if return_dict:
        return face_utilities_dict


def save_faces(image_path, face_numbers=None, database='faces.db'):
    face_utilities_dict = find_face_embeddings(image_path, report=False, return_dict=True)
    # copy dictionary keys
    # TODO: did not work! > selected_faces_dict = dict.fromkeys(face_utilities_dict.keys(), [])
    selected_faces_dict = {}
    for key in face_utilities_dict.keys():
        selected_faces_dict[key] = []
    # convert face_numbers to a list if it is not a list
    if type(face_numbers) != type([]):
        face_numbers = [face_numbers]
    # copy selected faces only
    for face_number in face_numbers:
        for key in face_utilities_dict.keys():
            selected_faces_dict[key].append(face_utilities_dict[key][face_number])
    face_database.save(database, selected_faces_dict)
    print('faces ({}) in image ({}) were saved sucessfully!'.format(face_numbers, image_path))


def find_face_locations_webcam(video_path=0, model='onnx'):
    if model == 'onnx':
        # takes 0.04 sec to load:
        onnx_path = "utilities/models/onnx/fixed_version-RFB-640.onnx"
        predictor = onnx.load(onnx_path)
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        threshold = 0.8

    cap = cv2.VideoCapture(video_path)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    process_time = []
    i = 0

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        start_time = time.time()

        if model == 'onnx':
            processed_frame = onnx_image_preprocessing(frame)
            confidences, boxes = ort_session.run(None, {input_name: processed_frame})
            face_locations, labels, probs = onnx_predict(frame.shape[1], frame.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            labels_probs = [(f"face({probs[i]:.2f})") for i in range(len(face_locations))]
        elif model == 'hog':
            face_locations = face_recognition.face_locations(frame)
            labels_probs = [(f"face(1.00)") for i in range(len(face_locations))]

        face_embeddings = face_recognition.face_encodings(frame, known_face_locations=face_locations, model='small')
        face_labels = classify_faces(face_embeddings)

        process_time.append(round(time.time() - start_time, 2))

        report_values = ('', process_time[i], labels_probs)

        labeled_frame = face_draw.draw_face_locations(image=frame, face_locations=face_locations, report_values=report_values,
                                                      lib='pil',
                                                      show_images=False, save_images=False, return_images=True,
                                                      label_faces=True, show_axes=False, show_points=False)

        cv2.imshow('Input', labeled_frame)
        i += 1

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Avg Proc Time [sec]: ', round(np.mean(process_time), 2))


def train_face_recognition_classifier(database_path='faces.db'):
    # recall embeddings from the database
    face_database_dict = face_database.retrieve(database_path)
    print('faces data base was retrieved')

    # train the model on these face embeddings

    print('face recognition classifier was trained successfully!')


def classify_faces(face_embeddings, method='direct'):  # or recognize face

    face_labels = []

    if method == 'direct':
        for face_embedding in face_embeddings:
            pass
    elif method == 'classifier':
        pass

    return face_labels


'''
# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
'''
