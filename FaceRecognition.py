from utilities import face_detect, face_database


def find_face_locations(images_folder, lib, save_dict):
    face_detect.detect_faces_in_images(images_folder,
                                       model='onnx',
                                       lib=lib,
                                       report=True,
                                       show_images=True,
                                       save_images=True,
                                       label_faces=True,
                                       classify_faces=True,
                                       show_landmarks=False,
                                       save_face_dict=save_dict)


def find_face_embeddings(images_folder):
    face_utilities_dict = face_detect.find_face_embeddings(images_folder, report=True, show_images=True, return_dict=True)
    #face_database_utilities.create('faces.db')
    #face_database_utilities.save('faces.db', face_utilities_dict)
    #print(face_utilities_dict)

if __name__ == '__main__':
    #face_database.delete(database_path='database/faces.db', to_delete='all')
    #print_database_dict('database/faces.db')
    #find_face_locations('input/imgs/1.jpg', lib='pil')
    #find_face_locations('input/imgs', lib='pil', save_dict=True)
    face_detect.detect_faces_in_videos(model='onnx', lib='pil', classify_faces=False, save_face_dict=False)
    #face_detect.detect_faces_in_videos(video_path='input/videos/lunch_scene.mp4',model='onnx', lib='pil', classify_faces=False, show_landmarks=False)
    #print_database_dict('database/faces.db')
    #face_detect.save_faces_dict(image_path='input/imgs/1.jpg', label_option='predefined')
    #print_database_dict('database/faces.db')


    #check_database_inputs_and_inputs('input/imgs/2.jpg')

    #print_database_dict('database/faces.db')
    #find_face_embeddings('imgs/2.jpg')
    #save_faces_dict('imgs/2.jpg', face_numbers=2)
    #face_database.delete(database_path='database/faces.db',to_delete='all')#,image_path='imgs/2.jpg')
    #print_database_dict('database/faces.db')
    #face_database_utilities.delete(database_path='database/faces.db', to_delete='face', image_path='imgs/24.jpg', face='24.jpg/face_0')


    #face_utilities.train_face_classifier()

    #face_detect.detect_faces_in_videos(model='onnx',lib='pil')
    #face_utilities.detect_faces_in_videos(video_path='videos/lunch_scene.mp4',model='onnx')

    #face_utilities.train_face_classifier()
