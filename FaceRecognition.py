from utilities import face_utilities



if __name__ == '__main__':

        # face_utilities.find_face_landmarks(image_path, enumerate_faces=True, report=True)
    # detect_imgs_onnx.find_face_locations(image_path, enumerate_faces=True, report=True)

    # Analyse folder
    #images_folder = 'imgs'
    #face_locations = face_utilities.find_face_locations_onnx(images_folder, lib='pil', report=True, show_images=True, save_images=False)
    #print(face_locations)

    image = 'imgs/2.jpg'
    face_embeddings = face_utilities.find_face_embeddings(image, show_images=False, save_embeddings=True)



