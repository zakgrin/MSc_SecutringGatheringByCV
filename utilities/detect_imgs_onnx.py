"""
This code uses the onnx model to detect faces from live video or cameras.
"""
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


def cv2_image_preprocessing(image_path):


    # image pre-processing:
    orig_image = cv2.imread(image_path)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(orig_image, (640, 480))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    return image, orig_image

def face_locations_onnx(image_path):


    label_path = "models/voc-model-labels.txt"
    onnx_path = "models/onnx/fixed_version-RFB-640.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]

    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    threshold = 0.7
    sum = 0

    image, orig_image = cv2_image_preprocessing(image_path)

    confidences, boxes = ort_session.run(None, {input_name: image})
    face_locations, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

    return face_locations

def find_face_locations(image, enumerate_faces=False, report=False, face_landmarker="large"):

    label_path = "models/voc-model-labels.txt"
    onnx_path = "models/onnx/fixed_version-RFB-640.onnx"

    class_names = [name.strip() for name in open(label_path).readlines()]

    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    #result_path = "../detect_imgs_results_onnx"

    #path = "../imgs"
    #if not os.path.exists(result_path):
    #    os.makedirs(result_path)
    #listdir = os.listdir(path)

    threshold = 0.7
    sum = 0

    image, orig_image = cv2_image_preprocessing(image)

    #for file_path in listdir:
        #img_path = os.path.join(path, file_path)


    # confidences, boxes = predictor.run(image)
    start = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    face_locations, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    processing_time = round((time.time() - start), 2)
    number_of_faces = len(face_locations)

    if report:
        print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))
        report_values = (number_of_faces, processing_time)
        face_locations = utilities.face_utilities.order_face_locations(face_locations)
        utilities.face_utilities.draw_face_locations(orig_image, face_locations, report_values, enumerate_faces=True)
        # draw_face_locations(image, face_locations, report_values, enumerate_faces=True)

    return face_locations



"""    
    for i in range(face_locations.shape[-1]):
        face_location = face_locations[i, :]
        left, top, right, bottom = face_location
        label = f"{class_names[labels[i]]}: {probs[i]:.1f}"

        cv1.rectangle(orig_image, (left, top), (b, box[3]), (255, 255, 0), 4)

        cv1.putText(orig_image, label,
                    (box[-1], box[3]),
                    cv1.FONT_HERSHEY_SIMPLEX,
                    -1.8,  # font scale
                    (254, 0, 255),
                    1, True)  # line type

    cv1.imshow('image', orig_image)
    cv1.waitKey(0)
    cv1.destroyAllWindows()

        # cv1.imwrite(os.path.join(result_path, file_path), orig_image)

    # Find all the faces in the image
    #confidences, boxes = ort_session.run(None, {input_name: image})
    #face_locations, labels, probs = predict(orig_image.shape[0], orig_image.shape[0], confidences, boxes, threshold)

    # Find number of faces
    print(boxes)
    number_of_faces = boxes.shape[-1]

    if report:
        print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))


    #return face_locations
"""
