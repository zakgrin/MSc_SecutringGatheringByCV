import argparse
from operator import itemgetter
from utilities import face_database, face_detect


def print_option(database, select=None):
    faces_dict_database = face_database.retrieve(database_path=database)
    n_faces = len(faces_dict_database['face'])
    if n_faces == 0:
        print('Face database is empty!')
    else:
        print('There are {} faces saved in the database ({}) as following:'.format(n_faces, database))
        if select in ['a', 'all']:
            print(faces_dict_database)
        elif select in ['face', 'path', 'location', 'embeddings']:
            print(faces_dict_database[select])
        else:
            print('Error: unknown argument ({}) for print option!'.format(arg))


def delete_option(database, all=False, image=None, face=None):
    if all:
        face_database.delete(database_path=database, to_delete='all')
    elif image and not face:
        face_database.delete(database_path=database, to_delete='image', image_path=image)
    elif face and not image:
        face_database.delete(database_path=database, to_delete='face', face=face)
    elif face and image:
        face_database.delete(database_path=database, to_delete='face', image_path=image, face=face)
    else:
        print('No delete option was selected! (options: --all, --image, --face)')


def save_option(database, label_opt='select', image=None, faces=None):
    if faces:
        face_detect.save_faces_dict(database=database, label_option='predefined',
                                    image_path=image, face_numbers=faces)
    else:
        face_detect.save_faces_dict(database=database, label_option=label_opt,
                                    image_path=image, face_numbers=faces)


def check_option(database, image_path):
    face_dict = face_detect.detect_faces_in_images(image_path, show_images=False, report=False, return_option='dict')
    face_database.save(database_path=database, face_utilities_dict=face_dict)
    face_dict_database_all = face_database.retrieve(database_path=database)
    indices = [i for i, x in enumerate(face_dict_database_all['path']) if x == image_path]
    face_dict_database = {}
    # test: check types and length of both dictionaries are the same
    for key in face_dict.keys():
        face_dict_database[key] = list(itemgetter(*indices)(face_dict_database_all[key]))
        assert len(face_dict[key]) == len(face_dict_database[key])
        assert type(face_dict[key]) == type(face_dict_database[key])
        assert type(face_dict[key][0]) == type(face_dict_database[key][0])
        print('Key="{}" was successfully stored and retrieved'.format(key))
    face_database.delete(database_path=database, to_delete='image', image_path=image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # define subparser
    subparser = parser.add_subparsers(help='Options:', dest='option')
    # database path (used with all options)
    parser.add_argument('-db', '--database', default='database/faces.db', metavar='',
                        help="Face database path")
    # print option
    print_parser = subparser.add_parser('print', aliases=['p'], help='print',
                                        formatter_class=argparse.RawTextHelpFormatter)
    print_parser.add_argument('-s', '--select', default='face',  metavar='',
                              help="select key to print:"
                                   "\n\t- all: print all values"
                                   "\n\t- face: print face names"
                                   "\n\t- locations: print face locations"
                                   "\n\t- embeddings: print face embeddings"
                                   "\n\t(default: face)")
    # delete option
    delete_parser = subparser.add_parser('delete', aliases=['d'], help='delete',
                                        formatter_class=argparse.RawTextHelpFormatter)
    delete_parser.add_argument('-a', '--all', default=False, action='store_true',
                               help="delete all faces in a database")
    delete_parser.add_argument('-i', '--image', default=None, type=str, metavar='',
                               help="delete all faces in a specific image")
    delete_parser.add_argument('-f', '--face', default=None, type=str, metavar='',
                               help="delete a specific face"
                                    "\n(in specific image if image was specified with face)")
    # save option
    save_parser = subparser.add_parser('save', aliases=['s'], help='save',
                                        formatter_class=argparse.RawTextHelpFormatter)
    save_parser.add_argument('-l', '--label', default='select', type=str, metavar='',
                             help="select face based on the following:"
                                  "\n- predefined: enter faces numbers (used with faces)"
                                  "\n- all: all faces in an image will be saved"
                                  "\n- select: selected faces defined by user"
                                  "\n- unselect: unselected faces defined by user"
                                  "\n(default: 'select')")
    save_parser.add_argument('-i', '--image', default=None, type=str, metavar='',
                             help="save faces in a specific image based on label option")
    save_parser.add_argument('-f', '--faces', default=None, type=int, metavar='', nargs='+',
                             help="save a specific face(s) based on detector number"
                                  "\n(default: None)")
    # check option
    check_parser = subparser.add_parser('check', aliases=['c'], help='chek')
    check_parser.add_argument('image', default=None, type=str,
                              help='an image file to check database with')
    # arguments
    args = parser.parse_args()
    if args.option in ['print', 'p']:
        print_option(database=args.database, select=args.select)
    elif args.option in ['delete', 'd']:
        delete_option(database=args.database, all=args.all, image=args.image, face=args.face)
    elif args.option in ['save', 's']:
        save_option(database=args.database, label_opt=args.label, image=args.image, faces=args.faces)
    elif args.option in ['check', 'c']:
        check_option(database=args.database, image_path=args.image)