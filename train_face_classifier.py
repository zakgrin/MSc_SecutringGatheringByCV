# USAGE
# python train_face_classifier.py
# -db faces.db -r output/recognizer.pickle -e output/encoder.pickle

# import the necessary packages
import argparse
import pickle
from utilities import face_database
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def trian_face_classifier(database_path: str, encoder_path: str, recognizer_path: str):

    # load the face embeddings
    print("[INFO] loading face embeddings ...")
    #data = pickle.loads(open(args["embeddings"], "rb").read())
    face_database_dict = face_database.retrieve(database_path)
    print(face_database_dict['face'])

    # encode the labels
    print("[INFO] encoding labels ...")
    le = LabelEncoder()
    #labels = le.fit_transform(data["names"])
    # TODO: here we have to find a way to balance the labels
    labels = le.fit_transform(face_database_dict["face"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model ...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(face_database_dict["embedding"], labels)

    # write the actual face recognition model to disk
    f = open(recognizer_path, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(encoder_path, "wb")
    f.write(pickle.dumps(le))
    f.close()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--database", required=True,
                    help="path to database")
    parser.add_argument("-r", "--recognizer", required=True,
                    help="path to output model trained to recognize faces")
    parser.add_argument("-e", "--encoder", required=True,
                    help="path to output label encoder")

    #args = vars(ap.parse_args()) # as dictionary
    args = parser.parse_args()

    trian_face_classifier(database_path=args.database,
                          recognizer_path=args.recognizer,
                          encoder_path=args.encoder)