import sqlite3 as lite
import numpy as np
from datetime import datetime


def create(database_path: str):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        cur.execute("drop table if exists faces")
        ddl = """CREATE TABLE faces (datetime TIMESTAMP,
                                     face TEXT PRIMARY KEY NOT NULL,
                                     path TEXT NOT NULL,
                                     location BLOB NOT NULL,
                                     embedding BLOB NOT NULL);"""
        cur.execute(ddl)
        ddl = "CREATE UNIQUE INDEX  faces_face_uindex ON faces (face)"
        cur.execute(ddl)
    conn.close()


def save(database_path: str, face_utilities_dict: dict):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        i = 0
        for face in face_utilities_dict['face']:
            # check to see if the word is in there
            sql = "SELECT count(face) FROM faces WHERE face='" + face + "' "
            cur.execute(sql)
            count = cur.fetchone()[0]

            if count > 0:
                sql = """UPDATE faces 
                         SET datetime='{0}',
                             path='{2}',
                             location='{3}',
                             embedding='{4}'
                         WHERE face='{1}'""".format(datetime.now(),
                                                    face_utilities_dict['face'][i],
                                                    face_utilities_dict['path'][i],
                                                    face_utilities_dict['location'][i],
                                                    face_utilities_dict['embedding'][i])
            else:
                sql = """INSERT INTO faces(datetime,face,path,location,embedding)
                         VALUES ('{0}','{1}','{2}','{3}','{4}')""".format(datetime.now(),
                                                                          face_utilities_dict['face'][i],
                                                                          face_utilities_dict['path'][i],
                                                                          face_utilities_dict['location'][i],
                                                                          face_utilities_dict['embedding'][i])
            cur.execute(sql)
            i += 1
    conn.close()
    print("Database saving operation is complete!")


def retrieve(database_path: str):
    face_database_dict = {'datetime': [], 'face': [], 'path': [], 'location': [], 'embedding': []}
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        sql = "SELECT * FROM faces"  # ORDER BY face DESC"
        cur.execute(sql)
        # TODO: initiating the dictionary in this way causes unworkable keys!
        # keys = [str(name[0]) for name in cur.description]
        # face_database_dict = dict.fromkeys(keys, [])
        for row in cur.fetchall():
            face_database_dict['datetime'].append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"))
            face_database_dict['face'].append(row[1])
            face_database_dict['path'].append(row[2])
            face_database_dict['location'].append([np.int32(loc) for loc in row[3].strip('[]').split(',')])
            face_database_dict['embedding'].append(np.array([np.float64(emb) for emb in
                                                             row[4].strip('[]').split(' ') if emb]))
    conn.close()
    return face_database_dict


def delete(database_path: str, to_delete: str, image_path: str = None, face: str = None):
    if to_delete == 'all':
        q = input('are you sure you want to delete all faces in database ({}) [y/n]: '.format(database_path))
        if q.lower() == 'y':
            create(database_path)
            print('all faces were deleted from the database ({})'.format(database_path))
        else:
            print('deletion process was canceled!')
    elif to_delete == 'image':
        if image_path is not None:
            # to delete specific all faces in image
            conn = lite.connect(database_path)
            with conn:
                cur = conn.cursor()
                sql = "DELETE FROM faces WHERE path='{0}'".format(image_path)
                cur.execute(sql)
            conn.close()
            print('all faces form image ({}) were deleted'.format(image_path))
        else:
            print('Error: image path is not provided')
    elif to_delete == 'face':
        if image_path is not None and face is not None:
            # to delete specific face in image
            conn = lite.connect(database_path)
            with conn:
                cur = conn.cursor()
                sql = "DELETE FROM faces WHERE path='{0}' AND face='{1}'".format(image_path, face)
                cur.execute(sql)
            conn.close()
            print('face ({}) from image ({}) was deleted'.format(face, image_path))
        elif image_path is None and face is not None:
            # to delete specific face in image
            conn = lite.connect(database_path)
            with conn:
                cur = conn.cursor()
                sql = "DELETE FROM faces WHERE face='{0}'".format(face)
                cur.execute(sql)
            conn.close()
            print('face ({}) was deleted'.format(face))
        else:
            print('Error: face or image path is not provided')
