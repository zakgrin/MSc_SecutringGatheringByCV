import face_recognition
import os
import re
import time
from datetime import datetime
import PIL
import cv2
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
import onnx
from utilities import face_database
from utilities import face_draw
from utilities import face_models

# Face detector
import utilities.vision.utils.box_utils_numpy as box_utils
# from caffe2.python.onnx import backend
# onnx runtime
import onnxruntime as ort

# Face recognizer (facenet: option 1)
from utilities.models.facenet1.inception_resnet_v1 import InceptionResNetV1


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


def onnx_image_preprocessing(image, size=(640, 480)):  # (320, 240)
    # image must be provided as RGB
    processed_image = cv2.resize(image, (size[0], size[1]))
    image_mean = np.array([127, 127, 127])
    processed_image = (processed_image - image_mean) / 128
    processed_image = np.transpose(processed_image, [2, 0, 1])
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image.astype(np.float32)
    return processed_image


def facenet_preprocessing(frame, face_locations=None):
    # image/frame must be provided as RGB > target size must be 160X160
    # https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
    faces = []
    if face_locations:
        # if we have face locations, then extract all faces
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            """
            To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to
            enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR
            (faster but still looks OK).
            """
            # Resize image:
            try:
                face_image = cv2.resize(face_image, (160, 160))  # , interpolation=cv2.INTER_CUBIC)
                faces.append(face_image)
            except cv2.error:
                faces.append(np.zeros((160, 160, 3), dtype='float32'))
    else:
        try:
            face_image = cv2.resize(frame, (160, 160))  # , interpolation=cv2.INTER_CUBIC)
            faces.append(face_image)
        except cv2.error:
            faces.append(np.zeros((160, 160, 3), dtype='float32'))
    faces = np.asarray(faces).astype('float32')  # or faces = np.stack(faces)
    mean, std = faces.mean(), faces.std()
    if std == 0: std = 1  # to avoid warning when std=0 due to cv2.error exception
    faces = (faces - mean) / std
    # expand_dims(face_pixels, axis=0)
    return faces


def detect_faces_in_images(images_folder, database='database/faces.db', detector='onnx', recognizer='facenet1',
                           lib='pil', trans=0.5, report=True, show_images=True, save_images=True,
                           show_axes=False, show_landmarks=False, save_face_dict=False,
                           return_option=None, classify_faces=False):

    # if option save_face_dict, then classify faces to check if they are registered or not
    if save_face_dict:
        classify_faces=True

    # load the detector if requested (0.04 sec to load)
    if detector == 'onnx':
        label_path = "utilities/models/voc-model-labels.txt"
        class_names = [name.strip() for name in open(label_path).readlines()]
        onnx_path = "utilities/models/onnx/fixed_version-RFB-640.onnx"
        # predictor = onnx.load(onnx_path)
        # onnx.checker.check_model(predictor)
        # onnx.helper.printable_graph(predictor.graph)
        # predictor = backend.prepare(predictor, device="CPU")  # default CPU
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        threshold = 0.7
        print('Face detector (onnx) was loaded (threshold={})'.format(threshold))

    # Face Recognition:
    if classify_faces:
        # Load face embeddings from the database
        face_dict_database = face_database.retrieve(database)
        face_embeddings_database = face_dict_database['embedding']
        similarity = 0.7

        # Load face recognizer model
        if recognizer == 'facenet1' or recognizer == 'facenet':
            recognizer_check = 'siamese'
            # https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/
            # https://github.com/serengil/tensorflow-101/blob/master/model/inception_resnet_v1.py
            facenet_model = InceptionResNetV1()
            # https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
            weights_path = "utilities/models/facenet1/facenet_weights.h5"
            facenet_model.load_weights(weights_path)
        elif classify_faces and recognizer == 'facenet2':
            recognizer_check = 'siamese'
            # https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
            # https://github.com/nyoki-mtl/keras-facenet
            facenet_path = "utilities/models/facenet2/facenet_keras.h5"
            facenet_model = keras.models.load_model(facenet_path, compile=False)
        else:
            recognizer = 'face_recognition'
            recognizer_check = 'siamese' # or 'compare'
            if recognizer_check != 'siamese':
                embeds_model = None
            similarity = 0.55 if recognizer_check == 'compare' else similarity
        print('face recognizer ({}) was loaded'.format(recognizer))

        # Load face similarity model
        if recognizer_check == 'siamese':
            embeds_shape = (128, 1)  # face_embeddings_database[0].shape
            embeds_model = face_models.init_siamse_model((embeds_shape), app='embeds', learn='after_l2')
            weights_path = "output/siamese_model/embeds_after_l2/" + recognizer + '_epoch_embeds_bs256_ep600_test_best.h5'
            embeds_model.load_weights(weights_path)
            print('face similarity model was loaded: {} for {}'.format(recognizer_check, recognizer))

    # to calc processing time
    process_time = []
    path_flag = True
    channels = 'RGB'
    # window specs
    if show_images:
        cv2.namedWindow('Output Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output Image', 1600, 900)
        cv2.moveWindow('Output Image', 0, 0)
    # check if it is an image array
    if isinstance(images_folder, np.ndarray):
        listdir = [images_folder]
        path_flag = False
        return_flag = True
        if report: print('an image array was provided')
    # if not an image, check if it is file path (not a folder path)
    elif not os.path.isdir(images_folder):
        image_file = os.path.basename(images_folder)
        images_folder = os.path.dirname(images_folder)
        listdir = [image_file]
        return_flag = True
        if report: print('an image path was provided')
    # if it is not an image array or file path, then it is a folder
    # give a list of files inside the folder but no return
    else:
        listdir = os.listdir(images_folder)
        return_flag = False
        return_option = None
        if report: print('a folder of images was provided')
    if path_flag:
        # creating output folder if it does not exist
        images_folder_result = "output" + re.split('input', images_folder)[1] + "_result"
        if not os.path.exists(images_folder_result):
            os.makedirs(images_folder_result)
    # values to report at for each file
    if report:
        print('-' * 60)
        print("{:<20s}{:>20s}{:>20s}".format('image-file', 'num-of-faces', 'process-time[sec]'))
        print('-' * 60)
    # creating embeddings dict if requested
    if return_option == 'dict' or save_face_dict:
        faces_dict = {'face': [], 'path': [], 'location': [], 'embedding': []}
    # loop through all files
    for image_file in listdir:
        if path_flag:
            # save image paths to read/save the image file
            image_path = os.path.join(images_folder, image_file)
            result_image_path = os.path.join(images_folder_result, image_file)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, dsize=(640, 480))
            label_option = 'select'
        else:
            image = image_file
            image_file = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')  # 'image'
            image_path = 'webcam'
            channels = 'BGR'
            label_option = 'all'
        # start processing time
        start_time = time.time()
        # calc face locations based on the selected detector
        if detector == 'onnx':
            processed_image = onnx_image_preprocessing(image)
            confidences, boxes = ort_session.run(None, {input_name: processed_image})
            face_locations, labels, probs = onnx_predict(image.shape[1], image.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            # labels_probs = [(f"{class_names[labels[i]]}({probs[i]:.2f})") for i in range(len(face_locations))]
        elif detector == 'hog':
            face_locations = face_recognition.face_locations(image, model='hog')
            # labels_probs = [f"face(?)" for i in range(len(face_locations))]

        # Create 'outsider' label for all detected faces
        labels_probs = ['outsider'] * len(face_locations)

        # calc face landmarks if requested
        if show_landmarks or return_option == 'landmarks':
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
        else:
            face_landmarks = None

        # calc face embeddings if requested
        if (return_option == 'embeddings' or return_option == 'dict' or classify_faces or save_face_dict) and \
                len(face_locations) > 0:

            if recognizer == 'facenet1' or recognizer == 'facenet2':
                faces = facenet_preprocessing(image, face_locations)
                face_embeddings = facenet_model.predict(faces)
                face_embeddings = [face_embedding for face_embedding in face_embeddings]  # .astype('float64')
            elif recognizer == 'face_recognition':
                face_embeddings = face_recognition.face_encodings(image, known_face_locations=face_locations,
                                                                  model='small')
            # compare faces' embeddings with database embeddings to find matches
            face_labels = compare_faces(database, face_embeddings, method=recognizer_check, embeds_model=embeds_model,
                                        similarity=similarity)
        else:
            face_labels = None

        # save face embeddings in a dict
        if return_option == 'dict' or save_face_dict:
            for i in range(len(face_embeddings)):
                faces_dict['face'].append(image_file + os.path.sep + 'face_' + str(i))
                faces_dict['path'].append(image_path)
                faces_dict['location'].append(face_locations[i])
                faces_dict['embedding'].append(face_embeddings[i])

        # save face dict in database:
        if save_face_dict:  # and not path_flag:
            save_faces_dict(image_path=image_path, label_option=label_option, faces_dict=faces_dict,
                            detector=detector, recognizer=recognizer)
            # update face labels, to include newly registered faces
            face_labels = compare_faces(face_embeddings_database, face_embeddings, method=recognizer_check,
                                        embeds_model=embeds_model, similarity=similarity)

        # add face calc processing time (locations, embeddings, landmarks)
        process_time.append(round(time.time() - start_time, 2))
        # print step report
        if report:
            print("{:<20s}{:>20d}{:>20.2f}".format(image_file, len(face_locations), process_time[-1]))
        # draw image with face locations if requested
        if show_images or save_images or show_axes:
            # create report dict for draw function
            report_dict = {'process_time': process_time[-1], 'labels_probs': labels_probs, 'channels': channels,
                           'face_labels': face_labels}
            # draw labeled image
            labeled_image = face_draw.face_locations(image=image,
                                                     report_dict=report_dict,
                                                     face_locations=face_locations,
                                                     face_landmarks=face_landmarks,
                                                     lib=lib,
                                                     trans=trans,
                                                     label_faces=classify_faces,
                                                     show_points=False,
                                                     show_landmarks=show_landmarks)
        # image output format
        if show_images:
            cv2.imshow('Output Image', labeled_image)
            cv2.waitKey(0)
        if show_axes:
            plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_images:
            labeled_image = cv2.resize(labeled_image, dsize=(640, 480))
            cv2.imwrite(result_image_path, labeled_image)
    # when finish close all windows (when imshow with different window names)
    if show_images:
        cv2.destroyAllWindows()
    # final report, print average processing time
    if report: print('Avg Proc Time [sec]: ', round(np.mean(process_time), 2))
    # if a single file, then return values based on return option
    if return_flag:
        if return_option == 'locations':
            if report: print('face locations are returned!')
            return face_locations
        elif return_option == 'embeddings':
            if report: print('face embeddings are returned!')
            return face_embeddings
        elif return_option == 'landmarks':
            if report: print('face landmarks are returned!')
            return face_landmarks
        elif return_option == 'dict':
            if report: print('face dict is returned!')
            return faces_dict
        else:
            return None


def load_face_recognizer(recognizer):
    pass


def detect_faces_in_videos(video_path=0, database='database/faces.db', detector='onnx', recognizer='facenet1',
                           lib='pil', trans=0.5, classify_faces=False, show_landmarks=False):
    # Load face detector model
    if detector == 'onnx':
        onnx_path = "utilities/models/onnx/fixed_version-RFB-640.onnx"
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        threshold = 0.6
        print('face detector {} was loaded'.format(detector))

    # Face Recognition:
    if classify_faces:
        # Load face embeddings from the database
        face_dict_database = face_database.retrieve(database)
        face_embeddings_database = face_dict_database['embedding']

        # Load face recognizer model
        if recognizer == 'facenet1' or recognizer == 'facenet':
            recognizer_check = 'siamese'
            similarity = 0.60
            # https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/
            # https://github.com/serengil/tensorflow-101/blob/master/model/inception_resnet_v1.py
            facenet_model = InceptionResNetV1()
            # https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
            weights_path = "utilities/models/facenet1/facenet_weights.h5"
            facenet_model.load_weights(weights_path)
        elif classify_faces and recognizer == 'facenet2':
            recognizer_check = 'siamese'
            similarity = 0.60
            # https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
            # https://github.com/nyoki-mtl/keras-facenet
            facenet_path = "utilities/models/facenet2/facenet_keras.h5"
            facenet_model = keras.models.load_model(facenet_path, compile=False)
        else:
            recognizer = 'face_recognition'
            recognizer_check = 'siamese' # or 'compare'
            if recognizer_check != 'siamese':
                embeds_model = None
            similarity = 0.55 if recognizer_check == 'compare' else 0.6
        print('face recognizer ({}) was loaded'.format(recognizer))

        # Load face similarity model
        if recognizer_check == 'siamese':
            embeds_shape = (128, 1)  # face_embeddings_database[0].shape
            embeds_model = face_models.init_siamse_model((embeds_shape), app='embeds', learn='after_l2')
            weights_path = "output/siamese_model/embeds_after_l2/" + recognizer + '_epoch_embeds_bs256_ep600_test_best.h5'
            embeds_model.load_weights(weights_path)
            print('face similarity model was loaded: {} for {}'.format(recognizer_check, recognizer))

    # Read video cap
    video_name = 'Webcam' if video_path == 0 else video_path
    cap = cv2.VideoCapture(video_path)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open the video: {}".format(video_name))

    # Set report and loop parameters
    process_time = []
    face_locations_previous = []
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video_name, 1600, 900)
    cv2.moveWindow(video_name, 0, 0)

    # Loop infinitely until the exit condition
    # - press 'Esc' to exit
    # - press 'Space' to register a face in face database
    while True:

        # Read a frame from the Webcam
        ret, frame = cap.read()
        #frame = cv2.resize(frame,dsize=(640, 480))

        # Start time
        start_time = time.time()

        # Calc face locations based on the selected face detector
        if detector == 'onnx':
            processed_frame = onnx_image_preprocessing(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            confidences, boxes = ort_session.run(None, {input_name: processed_frame})
            face_locations, labels, probs = onnx_predict(frame.shape[1], frame.shape[0], confidences, boxes, threshold)
            face_locations = onnx_order_face_locations(face_locations)
            # labels_probs = [(f"face({probs[i]:.2f})") for i in range(len(face_locations))]
        elif detector == 'hog':
            face_locations = face_recognition.face_locations(frame)
            # labels_probs = [(f"face(?)") for i in range(len(face_locations))]

        #        def track_faces(current_face_locations, previous_face_locations):
        #
        #            face_locations = [0]*len(current_face_locations)
        #            nearest_dist = [np.inf]*len(current_face_locations)
        #            if len(current_face_locations) > 1 and len(previous_face_locations) > 0:
        #                for i in range(len(current_face_locations)):
        #                    top_c, right_c, bottom_c, left_c = current_face_locations[i]
        #                    current_center = np.array([bottom_c-top_c, right_c-left_c])
        #                    distance_ref = np.inf
        #                    # nearest_index = 0
        #                    for j in range(len(previous_face_locations))
        #                        top_p, right_p, bottom_p, left_p = previous_face_locations[j]
        #                        previous_center = np.array([bottom_p-top_p, right_p-left_p])
        #                        distance = np.linalg.norm(current_center-previous_center)
        #                        if distance < distance_ref:
        #                            distance_ref = distance
        #                            nearest_index = j
        #                            nearest_dist = distance
        #                    if face_locations[nearest_index] == 0:
        #                        face_locations[nearest_index] = current_face_locations[i]
        #                    else:
        #                        # todo: we have to decide the colest between these two
        #                        current_center = current_face_locations[i]
        #                        previous_center = face_locations[nearest_index]
        #                        face_locations[nearest_index+1] = current_face_locations[i]
        #                    #previous_face_locations.pop(index_location)
        #                    #face_locations.append()
        #            else:
        #                return current_face_locations
        #
        #            return face_locations
        #
        #        # Track faces todo: this is just a tracker to order
        #        if not face_locations_previous:
        #            face_locations_previous = face_locations
        #        else:
        #            face_locations = track_faces(face_locations, face_locations_previous)
        #            face_locations_previous = face_locations

        # Create 'outsider' label for all detected faces
        labels_probs = ['outsider'] * len(face_locations)

        # Calc face landmarks if requested
        if show_landmarks:
            face_landmarks = face_recognition.face_landmarks(frame, face_locations)
        else:
            face_landmarks = None

        # Calc face embeddings if requested (to recognize faces)
        if classify_faces and len(face_locations) > 0:
            if recognizer == 'facenet1' or recognizer == 'facenet2':
                faces = facenet_preprocessing(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_locations)
                face_embeddings = facenet_model.predict(faces)
                face_embeddings = [face_embedding for face_embedding in face_embeddings]  # .astype('float64')
            elif recognizer == 'face_recognition':
                face_embeddings = face_recognition.face_encodings(frame, known_face_locations=face_locations,
                                                                  model='small')
            # compare faces' embeddings with database embeddings to find matches
            face_labels = compare_faces(face_embeddings_database, face_embeddings, method=recognizer_check,
                                        embeds_model=embeds_model, similarity=similarity)
        else:
            face_labels = None

        # Calc processing time
        process_time.append(round(time.time() - start_time, 2))

        # Label frame
        report_dict = {'process_time': process_time[-1], 'labels_probs': labels_probs, 'channels': 'BGR',
                       'face_labels': face_labels}
        labeled_frame = face_draw.face_locations(image=frame,
                                                 report_dict=report_dict,
                                                 face_locations=face_locations,
                                                 face_landmarks=face_landmarks,
                                                 lib=lib,
                                                 trans=trans,
                                                 label_faces=classify_faces,
                                                 show_points=False,
                                                 show_landmarks=show_landmarks)

        # Show the labeled frame
        cv2.imshow(video_name, labeled_frame)

        # Exit condition from the while loop
        c = cv2.waitKey(1)
        # press 'Esc' to exit
        if c == 27:
            break
        # press 'Space' to register a face in the database
        elif c == 32 and classify_faces and len(face_locations) > 0 and database is not None:
            faces_dict = {'face': [], 'path': [], 'location': [], 'embedding': []}
            name = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
            for i in range(len(face_embeddings)):
                faces_dict['face'].append(name + os.path.sep + 'face_' + str(i))
                faces_dict['path'].append('webcam')
                faces_dict['location'].append(face_locations[i])
                faces_dict['embedding'].append(face_embeddings[i])
            face_database.save(database, faces_dict)
            print('face embeddings were saved in the database ({})'.format(database))
            # Reload face embeddings from the database
            face_dict_database = face_database.retrieve(database)
            face_embeddings_database = face_dict_database['embedding']
        elif database is None:
            print('Error: please specify database path')
        else:
            continue

    # Release the cap and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Print average processing time
    print('Avg Proc Time [sec]: ', round(np.mean(process_time), 2))


def save_faces_dict(image_path, face_numbers=None, label_option='select', database='database/faces.db',
                    faces_dict=None, detector='onnx', recognizer='facenet1'):
    # todo: problems if a folder is provided!
    # label option: predefined
    if label_option == 'predefined' or label_option == 'all':
        # calc face dict for the image
        if faces_dict is None:
            faces_dict = detect_faces_in_images(image_path, detector=detector, recognizer=recognizer,
                                                lib='pil', report=False, show_images=False, save_images=False,
                                                show_axes=False, save_face_dict=False, show_landmarks=False,
                                                return_option='dict', classify_faces=False)
        if face_numbers is None and label_option != 'all':
            print('face_numbers argument was not defined, all faces will be saved!')
        elif face_numbers:
            # convert face_numbers to a list if it is not a list
            if not isinstance(face_numbers, list):
                face_numbers = [face_numbers]
            # edit index to match the dict
            face_numbers = [i - 1 for i in face_numbers]
        # save inputs flag
        if len(faces_dict['location']) > 0:
            inputs = 'y'
        else:
            inputs = 'n'
    # label option: selective
    elif label_option == 'select' or label_option == 'unselect':
        # if faces_dict was not provided, calc face dict for the image
        if faces_dict is None:
            faces_dict = detect_faces_in_images(image_path, detector=detector, recognizer=recognizer,
                                   lib='pil', report=False, show_images=True, save_images=False,
                                   show_axes=False, save_face_dict=False, show_landmarks=False,
                                   return_option='dict', classify_faces=False)
        # if faces_dict was provided, then just show the image with labels
        else:
            detect_faces_in_images(image_path, detector=detector, recognizer=recognizer,
                                   lib='pil', report=False, show_images=True, save_images=True,
                                   show_axes=False, save_face_dict=False, show_landmarks=False,
                                   return_option=None, classify_faces=True)
        num_faces = len(faces_dict['location'])
        if label_option == 'select':
            inputs = input('please select faces - list of face numbers (enter [n] to cancel): ')
            if inputs != 'n':
                face_numbers = [int(i) - 1 for i in re.split(r'\W+', inputs) if i is not '']
            # labels = [0] * len(num_faces)
            # for face_number in face_numbers:
            #    labels[face_number] = 1
        elif label_option == 'unselect':
            inputs = input('please unselect faces - list of face numbers: ')
            if inputs != 'n':
                unselected_face_numbers = [int(i) - 1 for i in re.split(r'\W+', inputs) if i is not '']
                face_numbers = [i for i in range(num_faces) if i not in unselected_face_numbers]
            # labels = [1] * len(num_faces)
            # for face_number in face_numbers:
            #    labels[face_number] = 0
    else:
        print('Error: label_option argument is not recognized!')
    # if image array then give a name
    if isinstance(image_path, np.ndarray):
        image_path = 'webcam'
    # save selected faces
    if face_numbers and inputs != 'n':
        # create empty dict for the selected faces
        selected_faces_dict = {}
        for key in faces_dict.keys():
            selected_faces_dict[key] = []
        # copy only selected faces from faces dict
        for face_number in face_numbers:
            for key in faces_dict.keys():
                selected_faces_dict[key].append(faces_dict[key][face_number])
        # save selected faces dict in database
        face_database.save(database, selected_faces_dict)
        print('faces ({}) in image ({}) were saved successfully!'.format([i + 1 for i in face_numbers], image_path))
        return True
    # cancel saving process
    elif inputs == 'n':
        print('no faces were saved from image ({})'.format(image_path))
        return False
    # if no face numbers then save all faces dict in database
    else:
        face_database.save(database, faces_dict)
        print('all faces in image ({}) were saved successfully!'.format(image_path))
        return True


def train_face_classifier(database_path='faces.db'):
    # recall embeddings from the database
    face_database_dict = face_database.retrieve(database_path)
    print('faces data base was retrieved')

    # train the detector on these face embeddings

    print('face recognition classifier was trained successfully!')


def step_func(dist, similarity):
    if dist >= similarity:
        return True
    return False


def compare_faces(face_embeddings_database, face_embeddings, method='siamese', embeds_model=None, similarity=0.55):  # best 0.55
    face_labels = []

    if len(face_embeddings_database) > 0 and method == 'siamese':
        if embeds_model is None:
            embeds_shape = face_embeddings_database[0].shape
            embeds_model = face_models.init_siamse_model((embeds_shape), api='embeds')
            # optimizer = keras.optimizers.Adam(lr=0.00006)
            # embeds_model.compile(loss="binary_crossentropy",optimizer=optimizer, metrics='accuracy')
            weights_path = "output/siamese_model/weights/epoch_embeds_bs256_ep680_test_best.h5"
            embeds_model.load_weights(weights_path)
            # model_path = "output/siamese_model/epoch_embeds_bs256_ep680_test_best.h5"
            # embeds_model = keras.models.load_model(model_path, compile=False)
        for face_embedding in face_embeddings:
            length = len(face_embeddings_database)
            h = face_embeddings[0].shape[0]
            shape = (length, h, 1)
            pairs = [np.zeros(shape, dtype=float) for i in range(2)]
            pairs[0] = np.tile(np.array(face_embedding), (length, 1)).reshape(shape)
            pairs[1] = np.array(face_embeddings_database).reshape(shape)
            probs = embeds_model.predict(pairs)
            #matches = [step_func(match, similarity) for match in probs]
            matches = [step_func(np.max(probs), similarity)]
            # name = "unknown"
            if True in matches:
                face_labels.append(1)
                # first_match_index = matches.index(True)
                # name = face_dict_database[first_match_index]
            else:
                face_labels.append(0)

    elif len(face_embeddings_database) > 0 and method == 'compare':
        #if method == 'compare':
        for face_embedding in face_embeddings:
            matches = face_recognition.compare_faces(face_embeddings_database, face_embedding,
                                                     tolerance=1 - similarity)
            # name = "unknown"
            if True in matches:
                face_labels.append(1)
                # first_match_index = matches.index(True)
                # name = face_dict_database[first_match_index]
            else:
                face_labels.append(0)
    #else:
        # no faces to check with
        #face_labels = []

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
	help="path to output detector trained to recognize faces")
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

# train the detector used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training detector...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition detector to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
'''
