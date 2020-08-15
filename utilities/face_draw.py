import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2


def face_locations(image, report_dict, face_locations, face_landmarks, lib='pil',
                   label_faces=False, show_points=False, show_landmarks=False):

    # information about faces and images
    number_of_faces = len(face_locations)
    process_time = report_dict['process_time']
    labels_probs = report_dict['labels_probs']
    channels = report_dict['channels']
    face_labels = report_dict['face_labels']

    if lib == 'pil':
        # if BGR: convert to RGB
        if channels.lower() == 'BGR'.lower():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        # create draw object
        draw = PIL.ImageDraw.Draw(image, 'RGBA')
        # draw text on top of the image
        font_list = ["arial.ttf", "handwriting-markervalerieshand-regular.ttf", "Drawing_Guides.ttf"]
        font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], 22)
        # transparancy
        trans = int(255*0.5)
        # loop in face locations
        for i in range(len(face_locations)):
            # check if person registered
            c = (255,0,0,trans) #"red"
            # If face_labels is not None, then convert Trues into green color with registered label
            if face_labels:
                if face_labels[i]:
                    c = (0,255,0,trans) #"green"
                    labels_probs[i] = 'registered'
            # draw face locations
            top, right, bottom, left = face_locations[i]

            #draw.rectangle([left, top, right, bottom], fill=c, outline=c, width=5)
            draw.rectangle([left, top, right, bottom], fill=(100,100,100,trans), width=5)
            rad = (right-left)*0.15
            draw.ellipse((right-rad, top-rad, right+rad, top+rad), fill=c) # top right
            # show vertex points for debugging
            if show_points:
                draw.ellipse((left - 5, top - 5, left + 5, top + 5), fill="yellow") # top left
                draw.ellipse((left - 5, bottom - 5, left + 5, bottom + 5), fill="orange") # bottom left
                draw.ellipse((right - 5, top - 5, right + 5, top + 5), fill="green") # top right
                draw.ellipse((right - 5, bottom - 5, right + 5, bottom + 5), fill="black") # bottom right
            # show labels below face locations
            if label_faces:
                # draw label box below face location
                bottom_ = bottom + int(0.2 * (bottom - top))
                #draw.rectangle([left, bottom, right, bottom_], fill=c, outline=c, width=5)
                draw.rectangle([left, bottom, right, bottom_], fill=c, width=5)
                # find text size based on the label box size
                text = "{}.{}".format(i + 1, labels_probs[i])
                face_font_size = 0
                face_font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], face_font_size)
                while face_font.getsize(text)[0] < right-left and face_font.getsize(text)[1] < bottom_-bottom:
                    face_font_size += 1
                    face_font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], face_font_size)
                face_font = PIL.ImageFont.truetype("input/fonts/" + font_list[0], abs(face_font_size-1))
                # put text on label box
                draw.text((left, bottom),
                          text,
                          fill=(0,0,0,trans),#'black'
                          font=face_font)
        # show faces landmarks
        if show_landmarks and face_landmarks:
            for i in range(len(face_landmarks)):
                for name, list_of_points in face_landmarks[i].items():
                    draw.line(list_of_points, fill=c, width=2)
        # summary on top of the image
        text = "Number of Faces = {} ({} seconds)".format(number_of_faces, process_time)
        width, height = font.getsize(text) # or draw.textsize(text, font=font)
        draw.rectangle([10, 10, width+10, height+10], fill=(255,255,255,trans)) #'white'
        draw.text((10, 10),
                  text,
                  fill=(0,0,0,trans), #'black'
                  font=font)
        # return to BGR
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    elif lib == 'cv2':
        # if RGB: convert to BGR
        if channels.lower() == 'RGB'.lower():
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # todo: add ovelay to include transparency in rectagle
        # code ..
        # loop in face locations
        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            cv2.rectangle(image, (left, top), (right, bottom), color=(0, 0, 255), thickness=4)
            # without ordering: (top, left), (bottom, right)
            # show vertex points for debugging
            if show_points:
                cv2.circle(image, (left, top), radius=5, thickness=-1, color=(0, 255, 255))  # top left yellow
                cv2.circle(image, (left, bottom), radius=5, thickness=-1, color=(0, 165, 255))  # bottom left orange
                cv2.circle(image, (right, top), radius=5, thickness=-1, color=(0, 255, 0))  # top right green
                cv2.circle(image, (right, bottom), radius=5, thickness=-1, color=(0, 0, 0))  # bottom right black
            # show labels below face locations
            if label_faces:
                # draw label box below face location
                bottom_ = bottom + int(0.15 * (bottom - top))
                cv2.rectangle(image, (left, bottom), (right, bottom_), (0, 0, 255), cv2.FILLED)
                # find text size based on the label box size
                text = "{}.{}".format(i + 1, labels_probs[i])
                face_font_size = 0.0
                width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, face_font_size, 2)[0]
                while width < right-left and height < bottom_-bottom:
                    face_font_size += 0.1
                    width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, face_font_size, 2)[0]
                face_font_size -= 0.1
                # put text on label box
                cv2.putText(image, text,
                            (left, int(bottom+(0.7*(bottom_-bottom)))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            face_font_size,
                            (255, 0, 0),
                            2)
        # show faces landmarks
        if face_landmarks and face_landmarks:
            for i in range(len(face_landmarks)):
                for name, list_of_points in face_landmarks[i].items():
                    cv2.drawContours(image, [np.array(list_of_points)], 0, (0,0,255),1)
        # summary on top of the image
        text = "Number of Faces = {} ({} seconds)".format(number_of_faces, process_time)
        width, height = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)[0]
        cv2.rectangle(image, (20, 5), (width+20, height+10), (255,255,255), cv2.FILLED)
        cv2.putText(image,
                    text,
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # font scale
                    (255, 0, 0),
                    2)

    # return as BGR
    return image

