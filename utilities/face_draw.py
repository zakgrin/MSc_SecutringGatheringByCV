import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2


def face_locations(image, face_locations, channels, report_dict, lib='pil', label_faces=False, show_points=False):

    number_of_faces = len(face_locations)
    process_time = report_dict['process_time']
    labels_probs = report_dict['labels_probs']

    if lib == 'pil':

        # if BGR: convert to RGB
        if channels.lower() == 'BGR'.lower():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = PIL.Image.fromarray(image)
        # Create draw object
        draw = PIL.ImageDraw.Draw(image)
        # Draw text on top of the image
        font_list = ["arial.ttf", "handwriting-markervalerieshand-regular.ttf", "Drawing_Guides.ttf"]
        font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], 22)
        if label_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, process_time)
            draw.text((10, 10),
                      text,
                      fill='blue',
                      font=font)

        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            draw.rectangle([left, top, right, bottom], outline="red", width=5)
            # show vertex points for debugging
            if show_points:
                draw.ellipse((left - 5, top - 5, left + 5, top + 5), fill="yellow") # top left
                draw.ellipse((left - 5, bottom - 5, left + 5, bottom + 5), fill="orange") # bottom left
                draw.ellipse((right - 5, top - 5, right + 5, top + 5), fill="green") # top right
                draw.ellipse((right - 5, bottom - 5, right + 5, bottom + 5), fill="black") # bottom right
            if label_faces:
                # TODO: here the text rectangle box has to be equal to all faces but as ration to adjust to face size
                draw.rectangle([left, bottom, right, bottom + int(0.3 * (bottom - top))],
                               fill="red", outline="red", width=5)
                text = "{}.{}".format(i + 1, labels_probs[i])
                draw.text((left, bottom),
                          text,
                          fill='blue',
                          font=font)

        # return to BGR
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    elif lib == 'cv2':

        # if RGB: convert to BGR
        if channels.lower() == 'RGB'.lower():
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if label_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, process_time)
            cv2.putText(image,
                        text,
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # font scale
                        (255, 0, 0),
                        2)

        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            cv2.rectangle(image, (left, top), (right, bottom), color=(0, 0, 255), thickness=5)
            # cv2.rectangle(image, (top, left), (bottom, right), color=(0, 0, 255), thickness=5) # without ordering
            if show_points:
                cv2.circle(image, (left, top), radius=5, thickness=-1, color=(0, 255, 255))  # top left yellow
                cv2.circle(image, (left, bottom), radius=5, thickness=-1, color=(0, 165, 255))  # bottom left orange
                cv2.circle(image, (right, top), radius=5, thickness=-1, color=(0, 255, 0))  # top right green
                cv2.circle(image, (right, bottom), radius=5, thickness=-1, color=(0, 0, 0))  # bottom right black
            if label_faces:
                text = "{}.{}".format(i + 1, labels_probs[i])
                cv2.putText(image, text,
                            (left, int(bottom + (0.2 * (bottom - top)))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

    # return as BGR
    return image


def face_landmarks(image, face_landmarks, report_values, lib='pil', show_images=True, save_images=True,
                   label_faces=False, show_axes=False, show_points=True):
    """

    :param label_faces: Show number of the face according to detection order
    """
    # report values
    number_of_faces = len(face_landmarks)
    if len(report_values) == 3:
        (image_path, processing_time, labels_probs) = report_values
    elif len(report_values) == 2:
        (image_path, processing_time) = report_values
        labels_probs = [f"face(?)" for i in range(number_of_faces)]

    if lib == 'pil' and (show_images or save_images):
        # Load the image into a Python Image Library object so that we can draw on top of it and display it
        pil_image = PIL.Image.fromarray(image)
        # Create draw object
        draw = PIL.ImageDraw.Draw(pil_image)
        # Draw text on top of the image
        font_list = ["arial.ttf", "handwriting-markervalerieshand-regular.ttf", "Drawing_Guides.ttf"]
        font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], 22)

        if label_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time)
            draw.text((10, 10),
                      text,
                      fill='blue',
                      font=font)

        for i in range(len(face_landmarks)):
            for name, list_of_points in face_landmarks[i].items():
                # Print the location of each facial feature in this image
                # print("The {} in this face has the following points: {}".format(name, list_of_points))
                draw.line(list_of_points, fill="red", width=2)

            if label_faces:
                left = min([x for x, y in list_of_points])
                bottom = max([y for x, y in list_of_points]) + (
                        max([y for x, y in list_of_points]) -
                        min([y for x, y in list_of_points]))
                # TODO: here the text rectangle boox has to be equal to all faces but as ration to adjust to face size
                draw.rectangle([left, bottom, left + 50, bottom + 50], fill="red", outline="red", width=5)
                print(i, len(labels_probs))
                text = "{}.{}".format(i + 1, labels_probs[i])
                draw.text((left, bottom),
                          text,
                          fill='blue',
                          font=font)

        # Display the image on screen
        if show_images:
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if show_axes:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_images and image_path != '':
            pil_image.save(image_path)

