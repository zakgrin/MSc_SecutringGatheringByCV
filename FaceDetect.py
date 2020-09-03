# Usage
# python FaceDetect.py v --path 0


import argparse
from utilities import face_detect


if __name__ == '__main__':
    # Define the main parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Database path (used with all options)
    parser.add_argument('-db', '--database', default='database/faces.db', metavar='',
                    help="Face database path"
                         "\n(default: 'database/faces.db')")

    # Add subparsers for options
    subparser = parser.add_subparsers(help='selection', dest='option')

    # Option 1: image
    image_parser = subparser.add_parser('image', aliases=['i'], formatter_class=argparse.RawTextHelpFormatter,
                                        help='image option')
    image_parser.add_argument('-p', '--path', default='input/imgs', type=str, metavar='',
                              help='image file or folder path')
    image_parser.add_argument('-m', '--detector', default='onnx', type=str, metavar='',
                              help="face detector ['onnx', 'hog']")
    image_parser.add_argument('-r', '--recognizer', default='face_recognition', type=str, metavar='',
                              help="face detector ['facenet', 'face_recognition'] \n(default: 'facenet')")
    image_parser.add_argument('-lib', '--library', default='pil', type=str, metavar='',
                              help="image annotation library ['pil', 'cv2'] \n(default: 'pil')")
    image_parser.add_argument('-tr', '--trans', default=0.5, type=float, metavar='',
                              help="annotation transparency \n(default: 0.5)")
    image_parser.add_argument('-rp', '--report', default=True, type=bool, metavar='',
                              help="print report [True, False]")
    image_parser.add_argument('-sh', '--show', default=True, type=bool, metavar='',
                              help="show image [True, False]")
    image_parser.add_argument('-s', '--save', default=False, action='store_true',
                              help="save processed images [True, False]")
#    image_parser.add_argument('-l', '--label', default=False, action='store_true',
#                              help="label faces [True, False]")
    image_parser.add_argument('-a', '--axes', default=False, action='store_true',
                              help="show axes [True, False]")
    image_parser.add_argument('-lm', '--landmarks', default=False, type=bool, metavar='',
                              help="show faces landmarks [True, False]")
    image_parser.add_argument('-d', '--save_dict', default=False, type=bool, metavar='',
                              help="save faces dictionary [True, False]")
    image_parser.add_argument('-ro', '--return_option', default='locations', type=str, metavar='',
                              help="classify faces [True, False]")
    image_parser.add_argument('-c', '--classify', default=False,  action='store_true',
                              help="classify faces [True, False]")

    # Option 2: video
    video_parser = subparser.add_parser('video', aliases=['v'], formatter_class=argparse.RawTextHelpFormatter,
                                        help="video option"
                                             "\n(press 'space' to register a face)")
    video_parser.add_argument('-p', '--path', default='0', metavar='',
                              help="video file path or IP address "
                                   "\n(option: path='ip' for IP: http://192.168.0.101:8080/video"
                                   "\n(default: path=0: Webcam)")
    video_parser.add_argument('-m', '--detector', default='onnx', type=str, metavar='',
                              help="face detector ['onnx', 'hog'] \n(default: 'onnx')")
    video_parser.add_argument('-r', '--recognizer', default='facenet2', type=str, metavar='',
                              help="face detector ['face_recognition', 'facenet1', 'facenet2'] \n(default: 'facenet1')")
    video_parser.add_argument('-lib', '--library', default='pil', type=str, metavar='',
                              help="image annotation library ['pil', 'cv2'] \n(default: 'pil')")
    video_parser.add_argument('-tr', '--trans', default=0.5, type=float, metavar='',
                              help="annotation transparency \n(default: 0.5)")
    video_parser.add_argument('-c', '--classify', default=False, action='store_true',
                              help="classify faces")
    video_parser.add_argument('-lm', '--landmarks', default=False, action='store_true',
                              help="show faces landmarks")

    args = parser.parse_args()
    if args.option in ['i', 'image', 'images']:
        face_detect.detect_faces_in_images(images_folder=args.path,
                                           database=args.database,
                                           detector=args.detector,
                                           recognizer=args.recognizer,
                                           lib=args.library,
                                           trans=args.trans,
                                           report=args.report,
                                           show_images=args.show,
                                           save_images=args.save,
                                           #label_faces=args.label,
                                           show_axes=args.axes,
                                           show_landmarks=args.landmarks,
                                           save_face_dict=args.save_dict,
                                           return_option=args.return_option,
                                           classify_faces=args.classify)

    elif args.option in ['v', 'video', 'videos']:
        if args.path == '0':
            args.path = 0
        elif args.path == 'ip':
            args.path = "http://192.168.0.101:8080/video"
            print('please make sure that webcam feeds are send to IP address: ', args.path)
        face_detect.detect_faces_in_videos(video_path=args.path,
                                           database=args.database,
                                           detector=args.detector,
                                           recognizer=args.recognizer,
                                           lib=args.library,
                                           trans=args.trans,
                                           classify_faces=args.classify,
                                           show_landmarks=args.landmarks)
