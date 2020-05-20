import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import face_recognition
import time
from utilities import detect_imgs_onnx

import os
import time
import PIL
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import utilities.face_utilities

# onnx runtime
import onnxruntime as ort


def find_face_locations(image_path, report=False, face_detector="ultra-light"):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param face_detector:   - "hog" is less accurate but faster on CPUs.
                            - "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available).
                            - The default is "hog".
    :return: face_locations
    """
    # Load the image as numpy array
    image = face_recognition.load_image_file(image_path)

    if report: start_time = time.time()

    # Find all the faces in the image
    if face_detector in ['hog','cnn']:
        face_locations = face_recognition.face_locations(image, model=face_detector)
    elif face_detector == 'ultra-light':
        face_locations = face_locations_onnx(image)

    if report: processing_time = round((time.time() - start_time), 2)

    # Find number of faces
    number_of_faces = len(face_locations)

    if report:
        print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))
        report_values = (number_of_faces, processing_time)
        if face_detector == 'ultra-light':
            face_locations = order_face_locations(face_locations)
        draw_face_locations(image, face_locations, report_values, lib='pil', enumerate_faces=True)

    return face_locations


def draw_face_locations(image, face_locations, report_values=None, lib='pil', enumerate_faces=False):
    """

    :param enumerate_faces: Show number of the face according to detection order
    """
    # report values
    if report_values:
        (number_of_faces, processing_time) = report_values

    if lib == 'pil':
        # Load the image into a Python Image Library object so that we can draw on top of it and display it
        pil_image = PIL.Image.fromarray(image)
        # Create draw object
        draw = PIL.ImageDraw.Draw(pil_image)
        # Draw text on top of the image
        font_list = ["arial.ttf", "handwriting-markervalerieshand-regular.ttf", "Drawing_Guides.ttf"]
        font = PIL.ImageFont.truetype("fonts/"+font_list[0], 22)

        if enumerate_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time)
            draw.text((10, 10),
                      text,
                      fill='blue',
                      font=font)

        i = 1
        for face_location in face_locations:
            # Print the location of each face in this image.
            # Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_location
            # Let's draw a box around the face
            draw.rectangle([left, top, right, bottom], outline="red", width=5)
            # TODO: here the text rectangle boox has to be equal to all faces but as ration to adjust to face size
            draw.rectangle([left, bottom, right, bottom+int(0.3*(bottom-top))], fill="red", outline="red", width=5)
            if enumerate_faces:
                text = "Face {}".format(i)
                draw.text((left, bottom),
                          text,
                          fill='blue',
                          font=font)
                i += 1

        # Display the image on screen
        pil_image.show()

    else: # cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if enumerate_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time)
            cv2.putText(image,
                        text,
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # font scale
                        (255, 0, 0),
                        2)

        i = 1
        for face_location in face_locations:
            # Print the location of each face in this image.
            # Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_location
            # Let's draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), color=(0, 0, 255), thickness=5)
            # cv2.rectangle(image, (top, left), (bottom, right), color=(0, 0, 255), thickness=5)
            if enumerate_faces:
                text = "Face {}".format(i)
                cv2.putText(image, text,
                            (left, int(bottom+(0.2*(bottom-top)))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)
                i += 1
        # Display the image on screen
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def order_face_locations(face_locations):
    # TODO: we may not need order!
    #stime = time.time()
    # for i in range(len(face_locations)):
    #     top, right, bottom, left = face_locations[i]
    #     face_locations[i] = [right, bottom, left, top]
    #period = time.time() - stime
    #print('ordering time', period)
    return [[right,bottom,left,top] for [top,right,bottom,left] in face_locations]


def find_face_landmarks(image, enumerate_faces=False, report=False, face_landmarker="large"):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param enumerate_faces: Show number of the face according to detection order
    :param face_landmarker: - "large" (default) or "small" which only returns 5 points but is faster
    :return:
    """

    image = load_image_file(image)
    # Find all the faces in the image
    start = time.time()
    face_landmarks_list = face_recognition.face_landmarks(image, model=face_landmarker)
    processing_time = round((time.time() - start), 2)
    # Find number of faces
    number_of_faces = len(face_landmarks_list)

    if report:
        print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)
    # Create draw object
    draw = PIL.ImageDraw.Draw(pil_image)
    # Draw text on top of the image
    #font = PIL.ImageFont.truetype("arial.ttf", 20)
    font = PIL.ImageFont.load_default()

    if enumerate_faces:
        draw.text((10, 10),
                  "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time),
                  fill='blue', font=font)

    #font = PIL.ImageFont.truetype("arial.ttf", 14)
    font = PIL.ImageFont.load_default()
    i = 1

    for face_landmarks in face_landmarks_list:
        for name, list_of_points in face_landmarks.items():
            # Print the location of each facial feature in this image
            # print("The {} in this face has the following points: {}".format(name, list_of_points))

            # Let's trace out each facial feature in the image with a line!
            draw.line(list_of_points, fill="red", width=2)
        if enumerate_faces:
            left = min([i for i, j in list_of_points])
            bottom = max([j for i, j in list_of_points]) + (
                    max([j for i, j in list_of_points]) -
                    min([j for i, j in list_of_points]))
            draw.text((left, bottom), "Face {}".format(i), fill='blue', font=font)
            i += 1

    # Display the image on screen
    pil_image.show()

    return face_landmarks_list



def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
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


def onnx_image_preprocessing(image, lib='cv2'):

    if lib == 'cv2':
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Add if there is a problem with the model predict
        processed_image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        processed_image = (processed_image - image_mean) / 128
        processed_image = np.transpose(processed_image, [2, 0, 1])
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = processed_image.astype(np.float32)

    elif lib == 'pil':
        processed_image = PIL.Image.fromarray(image)
        processed_image = processed_image.resize((640, 480))
        image_mean = np.array([127, 127, 127])
        processed_image = (processed_image - image_mean) / 128
        processed_image = np.transpose(processed_image, [2, 0, 1])
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = processed_image.astype(np.float32)

    return processed_image

def face_locations_onnx(image):

    # label_path = "models/voc-model-labels.txt"
    # class_names = [name.strip() for name in open(label_path).readlines()]

    onnx_path = "models/onnx/fixed_version-RFB-640.onnx"
    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    threshold = 0.7
    sum = 0

    processed_image = onnx_image_preprocessing(image, lib='cv2')

    confidences, boxes = ort_session.run(None, {input_name: processed_image})
    face_locations, labels, probs = predict(image.shape[1], image.shape[0], confidences, boxes, threshold)

    return face_locations

