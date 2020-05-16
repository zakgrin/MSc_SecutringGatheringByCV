import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import face_recognition
import time


def find_face_locations(image, enumerate_faces=False, report=False, face_detector="hog"):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param enumerate_faces: Show number of the face according to detection order
    :param face_detector:   - "hog" is less accurate but faster on CPUs.
                            - "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available).
                            - The default is "hog".
    :return: face_locations
    """
    # Find all the faces in the image
    start = time.time()
    face_locations = face_recognition.face_locations(image, model=face_detector)
    processing_time = round((time.time() - start), 2)
    # Find number of faces
    number_of_faces = len(face_locations)

    if report:
        print("I found {} face(s) in this photograph. in {} seconds".format(number_of_faces, processing_time))

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)
    # Create draw object
    draw = PIL.ImageDraw.Draw(pil_image)
    # Draw text on top of the image
    font = PIL.ImageFont.truetype("arial.ttf", 20)

    if enumerate_faces:
        draw.text((10, 10),
                  "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time),
                  fill='blue', font=font)

    font = PIL.ImageFont.truetype("arial.ttf", 14)
    i = 1

    for face_location in face_locations:
        # Print the location of each face in this image.
        # Each face is a list of co-ordinates in (top, right, bottom, left) order.
        top, right, bottom, left = face_location
        # Let's draw a box around the face
        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        if enumerate_faces:
            draw.text((left, bottom), "Face {}".format(i), fill='blue', font=font)
            i += 1
    # Display the image on screen
    pil_image.show()

    return face_locations


def find_face_landmarks(image, enumerate_faces=False, report=False, face_landmarker="large"):
    """
    :param report:
    :param image: An image (as a numpy array)
    :param enumerate_faces: Show number of the face according to detection order
    :param face_landmarker: - "large" (default) or "small" which only returns 5 points but is faster
    :return:
    """
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
    font = PIL.ImageFont.truetype("arial.ttf", 20)

    if enumerate_faces:
        draw.text((10, 10),
                  "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time),
                  fill='blue', font=font)

    font = PIL.ImageFont.truetype("arial.ttf", 14)
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

    return face_landmarks
