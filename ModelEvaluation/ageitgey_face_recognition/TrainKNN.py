############################################################################
#
#   Training a SVM for each of the Rooms
#
############################################################################

import os
import glob
import pickle
import argparse

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.calibration import CalibratedClassifierCV

import sau

#################################################################
#   Defining all the commandline arguments
#
ap= argparse.ArgumentParser()
ap.add_argument("-k", "--neighbors",
    help="No of Neighbors to concider",
    default=5
)
ap.add_argument("-w", "--weights",
    help="Enter a proper algorithm",
    default="uniform",
    const="uniform",
    nargs='?',
    choices=["uniform", "distance"]
)
ap.add_argument("-a", "--algorithm",
    help="Enter a proper algorithm",
    default="auto",
    const="auto",
    nargs='?',
    choices=["auto", "ball_tree", "kd_tree", "brute"]
)
args= vars(ap.parse_args())
#
#################################################################

if not os.path.exists(sau.trainedModdels_path):
    os.makedirs(sau.trainedModdels_path)

rooms= glob.glob(sau.registerData_path + "/*.room")

for room in rooms:
    room_name= room.split("\\")[-1].split('.')[0]
    print("[INFO - Workin on", room_name, "]")

    print("\t[INFO - LoadingEmbeedings]", "<location=", room, ">")
    room_serialized= pickle.loads(open(room, "rb").read())
    print("\t[INFO - LoadingSuccessful]")

    if(len(room_serialized["names"]) > 0):

        #################################################################
        #   Training the SVM Recognizer with Face Embeding Vectors
        #
        print("\t[INFO - TrainingSVM]", "<room=", room_name, ">")
        recognizer= KNeighborsClassifier(
            n_neighbors=int(args["neighbors"]),
            weights= args["weights"],
            algorithm= args["algorithm"],
            n_jobs= -1
        )
        # recognizer.fit(room_serialized["encodings"], room_serialized["names"])
        recognizer.fit(normalize(np.array(room_serialized["encodings"])), room_serialized["names"])
        print("\t[INFO - TraininSucessful]", "<room=", room_name, ">")
        #
        #################################################################

        #################################################################
        #   Storing the Trained Recognizer Model
        #
        recognizerFile_path= os.path.join(
            sau.trainedModdels_path,
            room_name + ".knn"
        )
        print("\t[INFO - StoringPredictor]", "<path=", recognizerFile_path, ">")
        reconizer_file= open(recognizerFile_path, "wb")
        reconizer_file.write(pickle.dumps(recognizer))
        reconizer_file.close()
        print("\t[INFO - StoringSuccessful]", "<file=", room_name+".knn", ">")
        #
        #################################################################
    else:
        print("\t[INFO - NoEmbedinPresent]")