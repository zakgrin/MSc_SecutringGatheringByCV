import sqlite3 as lite
import numpy as np
from datetime import datetime


def create_database(database_path: str):
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


def save_faces_to_data_base(database_path: str, face_embeddings_dict: dict):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        i=0
        for face in face_embeddings_dict['face']:
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
                                                  face_embeddings_dict['face'][i],
                                                  face_embeddings_dict['path'][i],
                                                  face_embeddings_dict['location'][i],
                                                  face_embeddings_dict['embedding'][i])
            else:
                sql = """INSERT INTO faces(datetime,face,path,location,embedding)
                         VALUES ('{0}','{1}','{2}','{3}','{4}')""".format(datetime.now(),
                                                                  face_embeddings_dict['face'][i],
                                                                  face_embeddings_dict['path'][i],
                                                                  face_embeddings_dict['location'][i],
                                                                  face_embeddings_dict['embedding'][i])
            cur.execute(sql)
            i+=1
    conn.close()
    print("Database operation is complete!")



def print_database_table(database_path: str):
    face_embeddings_dict = {'datetime':[],'face':[],'path':[],'location':[],'embedding':[]}
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        sql = "SELECT * FROM faces" #ORDER BY face DESC"
        cur.execute(sql)
        for row in cur.fetchall():
            face_embeddings_dict['datetime'].append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"))
            face_embeddings_dict['face'].append(row[1])
            face_embeddings_dict['path'].append(row[2])
            face_embeddings_dict['location'].append([np.int32(loc) for loc in row[3].strip('[]').split(',')])
            face_embeddings_dict['embedding'].append(np.array([np.float64(emb) for emb in
                                                               row[4].strip('[]').split(' ') if emb]))

            #face_embeddings_dict['location'].append(ast.literal_eval(row[2]))
            #face_embeddings_dict['embedding'].append(ast.literal_eval(row[3]))
            #face_embeddings_dict['location'].append(json.loads(row[2]))
            #face_embeddings_dict['embedding'].append(json.loads(row[3]))
            #face_embeddings_dict['location'].append(eval(row[2]))
            #face_embeddings_dict['embedding'].append(eval(row[3]))
    conn.close()
    return face_embeddings_dict
