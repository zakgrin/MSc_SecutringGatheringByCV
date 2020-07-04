import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2


def draw_face_locations(image, face_locations, report_values, lib='pil', show_images=True, save_images=False,
                        return_images=False, label_faces=False, show_axes=False, show_points=False):
    """

    :param label_faces: Show number of the face according to detection order
    """
    # report values
    # todo: change to dictionary!
    number_of_faces = len(face_locations)
    if len(report_values) == 3:
        (image_path, processing_time, labels_probs) = report_values
    elif len(report_values) == 2:
        (image_path, processing_time) = report_values
        labels_probs = [f"face(?)" for i in range(len(face_locations))]

    if lib == 'pil' and (show_images or save_images or return_images):

        if image_path == '':  # if Webcam then convert to pil (change channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        for i in range(len(face_locations)):
            # Print the location of each face in this image.
            # Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_locations[i]
            # Let's draw a box around the face
            draw.rectangle([left, top, right, bottom], outline="red", width=5)
            # show vertex points for debugging
            if show_points:
                # top left
                draw.ellipse((left - 5, top - 5, left + 5, top + 5), fill="yellow")
                # bottom left
                draw.ellipse((left - 5, bottom - 5, left + 5, bottom + 5), fill="orange")
                # top right
                draw.ellipse((right - 5, top - 5, right + 5, top + 5), fill="green")
                # bottom right
                draw.ellipse((right - 5, bottom - 5, right + 5, bottom + 5), fill="black")

            if label_faces:
                # TODO: here the text rectangle box has to be equal to all faces but as ration to adjust to face size
                draw.rectangle([left, bottom, right, bottom + int(0.3 * (bottom - top))],
                               fill="red", outline="red", width=5)
                text = "{}.{}".format(i + 1, labels_probs[i])
                draw.text((left, bottom),
                          text,
                          fill='blue',
                          font=font)

        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

        # Display the image on screen
        if show_images:
            ''' # slow !
            pil_image.show()
            os.wait()
            '''
            cv2.imshow('image', cv2_image)
            cv2.waitKey(0)
            # cv2.destroyAllWindows() # slow!
        if show_axes:
            plt.imshow(pil_image)  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_images and image_path != '':
            pil_image.save(image_path)
        if return_images:
            return cv2_image

    elif lib == 'cv2' and (show_images or save_images or return_images):

        if image_path != '':  # if not Webcam then convert to cv2 channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if label_faces:
            text = "Number of Faces = {} ({} seconds)".format(number_of_faces, processing_time)
            cv2.putText(image,
                        text,
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # font scale
                        (255, 0, 0),
                        2)

        for i in range(len(face_locations)):
            # Print the location of each face in this image.
            # Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_locations[i]
            # Let's draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), color=(0, 0, 255), thickness=5)
            # cv2.rectangle(image, (top, left), (bottom, right), color=(0, 0, 255), thickness=5) # without ordering
            # show vertex points for debugging
            if show_points:
                # top left
                cv2.circle(image, (left, top), radius=5, thickness=-1, color=(0, 255, 255))  # yellow
                # bottom left
                cv2.circle(image, (left, bottom), radius=5, thickness=-1, color=(0, 165, 255))  # orange
                # top right
                cv2.circle(image, (right, top), radius=5, thickness=-1, color=(0, 255, 0))  # green
                # bottom right
                cv2.circle(image, (right, bottom), radius=5, thickness=-1, color=(0, 0, 0))  # black
            if label_faces:
                text = "{}.{}".format(i + 1, labels_probs[i])
                cv2.putText(image, text,
                            (left, int(bottom + (0.2 * (bottom - top)))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

        # Display the image on screen
        if show_images:
            cv2.imshow('image', image)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if show_axes:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_images and image_path != '':
            cv2.imwrite(image_path, image)
        if return_images:
            return image


def draw_face_landmarks(image, face_landmarks, report_values, lib='pil', show_images=True, save_images=True,
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


def order_face_locations(face_locations):
    return [[right, bottom, left, top] for [top, right, bottom, left] in face_locations]
