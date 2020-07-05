from utilities import face_detect, face_database


def find_face_locations(images_folder):
    # face_utilities.find_face_landmarks(image_path, enumerate_faces=True, report=True)
    face_detect.detect_faces(images_folder,
                             model='hog',
                             lib='pil',
                             report=True,
                             show_images=True,
                             save_images=False,
                             label_faces=True)


def print_database_dict(database_path):
    face_database_dict = face_database.retrieve(database_path)
    print(face_database_dict['face'])

def check_database_inputs_and_inputs(image_path):
    # output: from face_utilities
    face_utilities_dict = face_detect.find_face_embeddings(image_path, show_images=False, return_dict=True)
    # input: from face_utilities to face_database_utilities
    face_database.create('database/faces.db')
    face_database.save('database/faces.db', face_utilities_dict)
    # output: from face_database_utilities
    face_database_dict = face_database.retrieve('database/faces.db')

    # test 1: check types and length of both dictionaries
    for key in face_utilities_dict.keys():
        assert len(face_utilities_dict[key]) == len(face_database_dict[key])
        assert type(face_utilities_dict[key]) == type(face_database_dict[key])
        assert type(face_utilities_dict[key][0]) == type(face_database_dict[key][0])
        print('Key="{}" was successfully stored and retrieved'.format(key))

    # test 2: check length of two dictionaries
    assert len(face_utilities_dict) + 1 == len(face_database_dict)
    print('test is successful!')


def find_face_embeddings(images_folder):
    face_utilities_dict = face_detect.find_face_embeddings(images_folder, report=True, show_images=True, return_dict=True)
    #face_database_utilities.create('faces.db')
    #face_database_utilities.save('faces.db', face_utilities_dict)
    #print(face_utilities_dict)

def save_faces(image_path,face_numbers):
    face_detect.save_faces(image_path, face_numbers)

if __name__ == '__main__':
    #find_face_locations('input/imgs')
    find_face_locations('input/imgs/3.jpg')
    #check_database_inputs_and_inputs('input/imgs/2.jpg')
    #print_database_dict('faces.db')
    #find_face_embeddings('imgs/2.jpg')
    #save_faces('imgs/2.jpg', face_numbers=2)
    #face_database_utilities.delete(database_path='faces.db',delete='image',image_path='imgs/2.jpg')
    #face_database_utilities.delete(database_path='faces.db', to_delete='face', image_path='imgs/24.jpg', face='24.jpg/face_0')

    #print_database_dict('faces.db')
    #face_utilities.train_face_recognition_classifier()

    #face_utilities.find_face_locations_webcam(model='onnx')
    #face_utilities.find_face_locations_webcam(video_path='videos/lunch_scene.mp4',model='onnx')

    #face_utilities.train_face_recognition_classifier()
