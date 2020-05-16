import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import face_recognition
import time


def find_face_locations(image, enamurate_faces=False, report=False, medel='hog'):
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
    face_locations = face_recognition.face_locations(image, model='hog')
    processing_time = round((time.time() - start), 2)
    # Find number of faces
    number_of_faces = len(face_locations)
    if report: print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))
    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)
    # Create draw object
    draw = PIL.ImageDraw.Draw(pil_image)
    # Draw text on top of the image
    font = PIL.ImageFont.truetype("arial.ttf", 20)
    if enamurate_faces: draw.text((10, 10),
                                  "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time),
                                  fill='blue', font=font)

    font = PIL.ImageFont.truetype("arial.ttf", 14)
    i = 1

    for face_location in face_locations:
        # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # Let's draw a box around the face
        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        if enamurate_faces:
            draw.text((left, bottom), "Face {}".format(i), fill='blue', font=font)
            i += 1
    # Display the image on screen
    pil_image.show()

    return face_locations


if __name__ == '__main__':
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("input/people.jpg")
    find_face_locations(image, enamurate_faces=True, report=True)
