############################################################################
#
#   Training a SVM for each of the Rooms
#
############################################################################

import os
import glob
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

registerData_path= "RegisteredData"

trainedModdel_path= "trainedSVM"
if not os.path.exists(trainedModdel_path):
    os.makedirs(trainedModdel_path)

rooms= glob.glob(registerData_path + "/*.room")

for room in rooms:
    room_name= room.split("\\")[-1].split('.')[0]
    print("[INFO - Workin on", room_name, "]")

    print("\t[INFO - LoadingEmbeedings]", "<location=", room, ">")
    room_serialized= pickle.loads(open(room, "rb").read())
    print("\t[INFO - LoadingSuccessful]")

    if(len(room_serialized["names"]) > 0):
        le= LabelEncoder()
        labels= le.fit_transform(room_serialized["names"])


        print("\t[INFO - TrainingSVM]", "<room=", room_name, ">")
        recognizer= SVC(C=1, kernel="linear", probability=True)
        recognizer.fit(room_serialized["encodings"], labels)
        print("\t[INFO - TraininSucessful]", "<room=", room_name, ">")

        recognizerFile_path= os.path.join(
            trainedModdel_path,
            room_name + ".svm"
        )
        print("\t[INFO - StoringPredictor]", "<path=", recognizerFile_path, ">")
        reconizer_file= open(recognizerFile_path, "wb")
        reconizer_file.write(pickle.dumps(recognizer))
        reconizer_file.close()
        print("\t[INFO - StoringSuccessful]", "<file=", room_name+".svm", ">")

        labelsFile_path= os.path.join(
            trainedModdel_path,
            room_name+".lables"
        )
        print("\t[INFO - StoringLabels]", "<path=", labelsFile_path, ">")
        labels_file= open(labelsFile_path, "wb")
        labels_file.write(pickle.dumps(le))
        labels_file.close()
        print("\t[INFO - StoringSuccessful]", "<file=", room_name+".lables", ">")
    else:
        print("\t[INFO - NoEmbedinPresent]")