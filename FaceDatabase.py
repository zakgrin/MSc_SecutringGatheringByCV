# USAGE
# python FaceDatabase.py --option print --argument face path

import argparse
from operator import itemgetter
from utilities import face_database, face_detect


def main(database: str, option: str, arguments: str, values=None):

    if option in ['p', 'print']:
        face_dict_database = face_database.retrieve(database_path=database)
        num = len(face_dict_database['face'])
        print('There are {} faces saved in the database ({}) as following:'.format(num, database))
        for arg in arguments:
            if arg in ['all', 'a']:
                print(face_dict_database)
            elif arg in ['face', 'f', 'path', 'p', 'location', 'l', 'embeddings', 'e']:
                print(face_dict_database[arg])
            else:
                print('Error: unknown argument ({}) for print option!'.format(arg))
    elif option in ['d', 'delete']:
        for arg in arguments:
            if arg in ['all', 'a']:
                face_database.delete(database_path=database, to_delete=arg)
                break
            elif arg in ['image', 'i']:
                for val in values:
                    face_database.delete(database_path=database, to_delete=arg, image_path=val)
            elif arg in ['face', 'f']:
                for val in values:
                    face_database.delete(database_path=database, to_delete=arg, face=val)
            elif arg in ['face_image', 'fi']:
                print('option', arg)
                path = values[1]
                face = values[0]
                face_database.delete(database_path=database, to_delete='face', image_path=path, face=face)
            else:
                print('Error: unknown argument ({}) for delete option!'.format(arg))
    elif option in ['s', 'save']:
        for arg in arguments:
            if arg in ['label', 'l']:
                index = arguments.index(arg)
                label_option = values[index]
            elif arg in ['image', 'i']:
                index = arguments.index(arg)
                image_path = values[index]
            elif arg in ['face', 'f']:
                index = arguments.index(arg)
                face_numbers = values[index]
            else:
                print('Error: unknown argument ({}) for save option!'.format(arg))
        # check options
        try: label_option
        except NameError: label_option = 'select'
        try: image_path
        except NameError: print('image path was not provided!')
        try: face_numbers
        except NameError: face_numbers = None
        # save face dict in database
        face_detect.save_faces_dict(database=database, label_option=label_option,
                                    image_path=image_path, face_numbers=face_numbers)
    elif option in ['c', 'check']:
        for arg in arguments:
            if arg in ['image', 'i']:
                index = arguments.index(arg)
                image_path = values[index]
            else:
                print('Error: unknown argument ({}) for check option!'.format(arg))
        face_dict = face_detect.detect_faces_in_images(image_path, show_images=False, report=False, return_option='dict')
        face_database.save(database_path=database, face_utilities_dict=face_dict)
        face_dict_database_all = face_database.retrieve(database_path=database)
        indices = [i for i, x in enumerate(face_dict_database_all['path']) if x == image_path]
        face_dict_database = {}
        # test: check types and length of both dictionaries
        for key in face_dict.keys():
            face_dict_database[key] = list(itemgetter(*indices)(face_dict_database_all[key]))
            assert len(face_dict[key]) == len(face_dict_database[key])
            assert type(face_dict[key]) == type(face_dict_database[key])
            assert type(face_dict[key][0]) == type(face_dict_database[key][0])
            print('Key="{}" was successfully stored and retrieved'.format(key))
        face_database.delete(database_path=database, to_delete='image', image_path=image_path)
    else:
        print('Error: unknown option!')


if __name__ == "__main__":
    # Setup the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-db', '--database', default='database/faces.db',
        help="Database path")
    parser.add_argument('-o', '--option', default='print',
        help="""Operation option
        - p or print: print database
        - d or delete: delete database
        - s or save: save face in database
        - c or check: check database
                """)
    parser.add_argument('-a', '--argument', default='face', nargs='+',
        help="""Operation argument
        - Option: p or print:
            - a or all: print database
            - f or face: print face names
            - p or path: print face paths
            - l or location: print face locations
            - e or embedding: print face embeddings
            - *multiple: (e.g. --argument face location ..)
        - Option: d or delete:
            - a or all: delete all faces
            - i or image: delete all faces with image path
            - f or face: delete specific face
            - fi or face_image: delete specific face in specific image 
        - Option: s or save: 
            - l or label: label option
            - i or image: image path
            - f or face: face number
        - Option: c or check:
            - i or image: image path 
                """)
    parser.add_argument('-v', '--value', default=None, nargs='+',
        help="""Argument value
        - Option: p or print:
            - no value is required
        - Option: d or delete:
            - Argument all: no value is required
            - Argument image: enter image path
            - Argument face: enter face name (use only if face has a unique name)
            - Argument face_image: enter face name and image path
                e.g. --option delete --argument face_image --value img1.jpg/face_1 input/imgs/img1.jpg
                (in this order, use to delete specific face in specific image)
        - Option: s or save:
            - Argument path: image path to save all faces in this image
            - Argument face: face numbers as list for specific image
            - Argument label: select face based on the following selection modes
                - predefined: enter faces numbers (used with face argument)
                - all: all faces will be saved
                - select: selected faces defined by user
                - unselect: unselected faces defined by user
                - if no option is selected, then 'select' is used as default.
        - Option: c or check:
            - Argument image: image path to check database with (it will be deleted from the database) 
                """)

    args = parser.parse_args()
    main(database=args.database, option=args.option, arguments=args.argument, values=args.value)
