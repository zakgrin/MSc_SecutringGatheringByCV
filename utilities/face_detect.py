import face_recognition
import os
import re
import time
# import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def onnx_image_preprocessing(image, size=(640, 480)): #(320, 240)
    processed_image = cv2.resize(image, (size[0], size[1]))
    image_mean = np.array([127, 127, 127])
    processed_image = (processed_image - image_mean) / 128
    processed_image = np.transpose(processed_image, [2, 0, 1])
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image.astype(np.float32)
    return processed_image


def detect_faces_in_images(images_folder, model='onnx', lib='pil', report=True, show_images=True, save_images=True,
                           label_faces=True, show_axes=False, show_landmarks=False, return_option='locations'):
    # load the model if requested (0.04 sec to load)
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
        print('Model onnx was loaded with threshold of {}'.format(threshold))
    # to calc processing time
    total_process_time = 0
    # check if folder of single file
    if not os.path.isdir(images_folder):
        image_file = os.path.basename(images_folder)
        images_folder = os.path.dirname(images_folder)
        listdir = [image_file]
        return_flag = True
    else:
        listdir = os.listdir(images_folder)
        return_flag = False
        return_option = None
    # creating output folder if it does not exist
    images_folder_result = "output" + re.split('input',images_folder)[1] + "_result"
    if not os.path.exists(images_folder_result):
        os.makedirs(images_folder_result)
    # values to report at for each file
    if report:
        print('-' * 60)
        print("{:<20s}{:>20s}{:>20s}".format('image-file', 'num-of-faces', 'process-time(sec)'))
        print('-' * 60)
    # creating embeddings dict if requested
    if return_option == 'embeddings':
        face_embeddings_dict = {'face': [], 'path': [], 'location': [], 'embedding': []}
    # loop through all files
    for image_file in listdir:
        start_time = time.time()
        # save image paths to read/save the image file
        image_path = os.path.join(images_folder, image_file)
        result_image_path = os.path.join(images_folder_result, image_file)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # calc face locations based on the selected model
        if model == 'onnx':
            processed_image = onnx_image_preprocessing(image)
            confidences, boxes = ort_session.run(None, {input_name: processed_image})
            face_locations, labels, probs = onnx_predict(image.shape[1], image.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            labels_probs = [(f"{class_names[labels[i]]}({probs[i]:.2f})") for i in range(len(face_locations))]
        elif model == 'hog':
            face_locations = face_recognition.face_locations(image, model='hog')
            labels_probs = [f"face(?)" for i in range(len(face_locations))]
        # calc face landmarks if the were requested
        if show_landmarks or return_option == 'landmarks' or return_option == 'all':
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
        else:
            face_landmarks = None
        # calc face embeddings if they were requested
        if return_option == 'embeddings' or return_option == 'all':
            face_embeddings = face_recognition.face_encodings(image, known_face_locations=face_locations, model='small')
            # save face embeddings in a dict
            for i in range(len(face_embeddings)):
                face_embeddings_dict['face'].append(image_file + os.path.sep + 'face_' + str(i))
                face_embeddings_dict['path'].append(image_path)
                face_embeddings_dict['location'].append(face_locations[i])
                face_embeddings_dict['embedding'].append(face_embeddings[i])
        # calc reporting parameters
        number_of_faces = len(face_locations)
        process_time = round((time.time() - start_time), 2)
        total_process_time += process_time
        # print step report
        if report:
            print("{:<20s}{:>20d}{:>20.2f}".format(image_file, number_of_faces, process_time))
        # draw image with face locations if requested
        if show_images or save_images or show_axes:
            # create report dict for draw function
            report_dict = {'process_time':process_time,'labels_probs':labels_probs, 'channels':'RGB'}
            # draw labeled image
            labeled_image = face_draw.face_locations(image=image,
                                                     report_dict=report_dict,
                                                     face_locations=face_locations,
                                                     face_landmarks=face_landmarks,
                                                     lib=lib,
                                                     label_faces=label_faces,
                                                     show_points=False,
                                                     show_landmarks=show_landmarks)
        # image output format
        if show_images:
            cv2.imshow('output image', labeled_image)
            cv2.waitKey(0)
        if show_axes:
            plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_images:
            cv2.imwrite(result_image_path, labeled_image)
    # when finish close all windows (when imshow with different window names)
    if show_images:
        cv2.destroyAllWindows()
    # final report
    if report:
        print("Total processing time (seconds): ", round(total_process_time, 2))
    # if a single file, then return values based on return option
    if return_flag:
        if return_option == 'locations':
            print('face locations are returned!')
            return face_locations
        elif return_option == 'embeddings':
            print('face embeddings are returned!')
            return face_embeddings_dict
        elif return_option == 'landmarks':
            print('face landmarks are returned!')
            return face_landmarks
        elif return_option == 'all':
            print('face locations, embeddings, landmarks are returned!')
            return face_locations, face_embeddings_dict, face_landmarks


def detect_faces_in_videos(video_path=0, model='onnx', lib='pil', classify_faces=False, show_landmarks=False):
    # load the model if requested (0.04 sec to load)
    if model == 'onnx':
        onnx_path = "utilities/models/onnx/fixed_version-RFB-640.onnx"
        predictor = onnx.load(onnx_path)
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        threshold = 0.8
    # read the video
    cap = cv2.VideoCapture(video_path)
    # check if the Webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open Webcam")
    # set report and loop parameters
    process_time = []
    i = 0
    # loop infinitely until the exit condition
    while True:
        # read a frame from the Webcam
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        start_time = time.time()
        # calc face locations based on the selected model
        if model == 'onnx':
            processed_frame = onnx_image_preprocessing(frame)
            confidences, boxes = ort_session.run(None, {input_name: processed_frame})
            face_locations, labels, probs = onnx_predict(frame.shape[1], frame.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            labels_probs = [(f"face({probs[i]:.2f})") for i in range(len(face_locations))]
        elif model == 'hog':
            face_locations = face_recognition.face_locations(frame)
            labels_probs = [(f"face(?)") for i in range(len(face_locations))]
        # calc face landmarks if the were requested
        if show_landmarks:
            face_landmarks = face_recognition.face_landmarks(frame, face_locations)
        else:
            face_landmarks = None
        # calc face embeddings if they were requested to recognize faces
        if classify_faces:
            face_embeddings = face_recognition.face_encodings(frame, known_face_locations=face_locations, model='small')
            face_labels = classify_faces(face_embeddings)
        # add
        process_time.append(round(time.time() - start_time, 2))

        report_dict = {'process_time':process_time[i],'labels_probs':labels_probs, 'channels':'BGR'}
        labeled_frame = face_draw.face_locations(image=frame,
                                                 report_dict=report_dict,
                                                 face_locations=face_locations,
                                                 face_landmarks=face_landmarks,
                                                 lib=lib,
                                                 label_faces=True,
                                                 show_points=False,
                                                 show_landmarks=False)
        # show the labeled frame
        cv2.imshow('Webcam', labeled_frame)
        i += 1
        # exit condition from the while loop
        c = cv2.waitKey(1)
        if c == 27:
            break
    # release the cap and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # print average processing time
    print('Avg Proc Time [sec]: ', round(np.mean(process_time), 2))


def save_faces(image_path, face_numbers=None, database='database/faces.db'):
    # calc face dict for the image
    faces_dict = detect_faces_in_images(image_path, model='onnx', lib='pil', report=True, show_images=False,
                                                  save_images=False, label_faces=False, show_axes=False,
                                                  show_landmarks=False, return_option='embeddings')
    # create empty dict for the selected faces
    selected_faces_dict = {}
    for key in faces_dict.keys():
        selected_faces_dict[key] = []
    # convert face_numbers to a list if it is not a list
    if type(face_numbers) != type([]):
        face_numbers = [face_numbers]
    # copy selected faces dict only
    for face_number in face_numbers:
        for key in faces_dict.keys():
            selected_faces_dict[key].append(faces_dict[key][face_number])
    # save selected faces dict in database
    face_database.save(database, selected_faces_dict)
    print('faces ({}) in image ({}) were saved successfully!'.format(face_numbers, image_path))


def train_face_classifier(database_path='faces.db'):
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
