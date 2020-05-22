from utilities import face_utilities



if __name__ == '__main__':

        # face_utilities.find_face_landmarks(image_path, enumerate_faces=True, report=True)
    # detect_imgs_onnx.find_face_locations(image_path, enumerate_faces=True, report=True)

    # Analyse folder
    images_folder = 'imgs'
    face_locations = face_utilities.find_face_landmarks_fr(images_folder, lib='pil', report=True, show_images=True, save_images=False)
    print(face_locations)

'''
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
'''



