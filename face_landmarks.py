import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import face_recognition
import time


def find_face_landmarks(image, enamurate_faces=False, report=False, medel='hog'):
    '''
    :param image: An image (as a numpy array)
    :param enamurate_faces: Show number of the face according to detection order
    :param medel:   - "hog" is less accurate but faster on CPUs.
                    - "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available).
                    The default is "hog".
    :return:
    '''
    # Find all the faces in the image
    start = time.time()
    face_landmarks_list = face_recognition.face_landmarks(image)
    processing_time = round((time.time() - start), 2)
    # Find number of faces
    number_of_faces = len(face_landmarks_list)
    if report: print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))
    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)
    # Create draw object
    draw = PIL.ImageDraw.Draw(pil_image)
    # Draw text on top of the image
    font = PIL.ImageFont.truetype("arial.ttf", 20)
    if enamurate_faces: draw.text((10, 10),"Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time),fill='blue',font=font)

    font = PIL.ImageFont.truetype("arial.ttf", 14)
    i = 1

    for face_landmarks in face_landmarks_list:
        for name, list_of_points in face_landmarks.items():
            # Print the location of each facial feature in this image
            #print("The {} in this face has the following points: {}".format(name, list_of_points))

            # Let's trace out each facial feature in the image with a line!
            draw.line(list_of_points, fill="red", width=2)

    # Display the image on screen
    pil_image.show()

    return face_landmarks

if __name__ == '__main__':
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("input/people.jpg")
    find_face_landmarks(image, enamurate_faces=True, report=True)