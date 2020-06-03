import cv2
from utilities import face_utilities
import face_recognition

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    box = face_utilities.find_face_locations_onnx_image(frame)
    #box = face_recognition.face_locations(frame)
    frame = face_utilities.draw_face_locations(image=frame, face_locations=box, report_values=['a','b','c'], lib='cv2',
                                               show_images=False, save_images=False, label_faces=True, show_axes=False,
                                               show_points=False, return_image=True)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()