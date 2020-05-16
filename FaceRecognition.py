from utilities import face_utilities
from face_recognition import load_image_file

if __name__ == '__main__':
    # Load the jpg file into a numpy array
    image = load_image_file("input/people.jpg")
    face_utilities.find_face_locations(image, enumerate_faces=True, report=True)
    face_utilities.find_face_landmarks(image, enumerate_faces=True, report=True)
