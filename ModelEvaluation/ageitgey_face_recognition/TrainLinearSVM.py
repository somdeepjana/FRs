############################################################################
#
#   Training a SVM for each of the Rooms
#
############################################################################

import os
import glob
import pickle
import argparse

from sklearn.svm import LinearSVC

import sau

#################################################################
#   Defining all the commandline arguments
#
ap= argparse.ArgumentParser()
ap.add_argument("-c", "--regularization",
    help="Regularization parameter. The strength of the regularization is inversely proportional to C",
    default=1.0
)
ap.add_argument("-r", "--randomstate",
    help="Give Random state Generator Nmber for constant Probability",
    default= 10
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
        recognizer= LinearSVC(
            C=float(args["regularization"]),
            random_state= int(args["randomstate"])
        )
        recognizer.fit(room_serialized["encodings"], room_serialized["names"])
        print("\t[INFO - TraininSucessful]", "<room=", room_name, ">")
        #
        #################################################################

        #################################################################
        #   Storing the Trained Recognizer Model
        #
        recognizerFile_path= os.path.join(
            sau.trainedModdels_path,
            room_name + ".linearsvm"
        )
        print("\t[INFO - StoringPredictor]", "<path=", recognizerFile_path, ">")
        reconizer_file= open(recognizerFile_path, "wb")
        reconizer_file.write(pickle.dumps(recognizer))
        reconizer_file.close()
        print("\t[INFO - StoringSuccessful]", "<file=", room_name+".linearsvm", ">")
        #
        #################################################################
    else:
        print("\t[INFO - NoEmbedinPresent]")