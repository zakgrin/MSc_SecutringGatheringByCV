import argparse
from utilities import face_detect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()#formatter_class=argparse.RawDescriptionHelpFormatter)
    subparser = parser.add_subparsers(help='selection', dest='option')
    # image parser
    image_parser = subparser.add_parser('image', aliases=['i'], help='image option')
    image_parser.add_argument('-p', '--path', default='input/imgs', type=str, metavar='',
                              help='image file or folder path')
    image_parser.add_argument('-m', '--model', default='onnx', type=str, metavar='',
                              help="  face detection model ['onnx', 'hog']")
    image_parser.add_argument('-lib', '--library', default='pil', type=str, metavar='',
                              help="image annotation library ['pil', 'cv2']")
    image_parser.add_argument('-r', '--report', default=True, type=bool, metavar='',
                              help="print report [True, False]")
    image_parser.add_argument('-sh', '--show', default=True, type=bool, metavar='',
                              help="show image [True, False]")
    image_parser.add_argument('-s', '--save', default=False, type=bool, metavar='',
                              help="save processed images [True, False]")
    image_parser.add_argument('-l', '--label', default=True, type=bool, metavar='',
                              help="label faces [True, False]")
    image_parser.add_argument('-a', '--axes', default=True, type=bool, metavar='',
                              help="show axes [True, False]")
    image_parser.add_argument('-lm', '--landmarks', default=False, type=bool, metavar='',
                              help="show faces landmarks [True, False]")
    image_parser.add_argument('-d', '--save_dict', default=False, type=bool, metavar='',
                              help="save faces dictionary [True, False]")
    image_parser.add_argument('-ro', '--return_option', default='locations', type=str, metavar='',
                              help="classify faces [True, False]")
    image_parser.add_argument('-c', '--classify', default=False, type=bool, metavar='',
                              help="classify faces [True, False]")
    # video parser
    video_parser = subparser.add_parser('video', aliases=['v'], help='video option')
    video_parser.add_argument('-p', '--path', default='input/videos/lunch_scene.mp4', metavar='',
                              help='video file path')
    video_parser.add_argument('-m', '--model', default='onnx', type=str, metavar='',
                              help="  face detection model ['onnx', 'hog']")
    video_parser.add_argument('-lib', '--library', default='pil', type=str, metavar='',
                              help="image annotation library ['pil', 'cv2']")
    video_parser.add_argument('-c', '--classify', default=False, type=bool, metavar='',
                              help="classify faces [True, False]")
    video_parser.add_argument('-lm', '--landmarks', default=False, type=bool, metavar='',
                              help="show faces landmarks [True, False]")
    video_parser.add_argument('-d', '--save_dict', default=False, type=bool, metavar='',
                              help="save faces dictionary [True, False]")

    args = parser.parse_args()
    if args.option in ['i', 'image', 'images']:
        face_detect.detect_faces_in_images(images_folder=args.path,
                                           model=args.model,
                                           lib=args.library,
                                           report=args.report,
                                           show_images=args.show,
                                           save_images=args.save,
                                           label_faces=args.label,
                                           show_axes=args.axes,
                                           show_landmarks=args.landmarks,
                                           save_face_dict=args.save_dict,
                                           return_option=args.return_option,
                                           classify_faces=args.classify)

    elif args.option in ['v', 'video', 'videos']:
        args.path = 0 if args.path == '0' else args.path
        face_detect.detect_faces_in_videos(video_path=args.path,
                                           model=args.model,
                                           lib=args.library,
                                           classify_faces=args.classify,
                                           show_landmarks=args.landmarks,
                                           save_face_dict=args.save_dict)
