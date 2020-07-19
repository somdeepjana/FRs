############################################################################
#
#   Testing the Recognition Manually Visual FeedBack
#
############################################################################

import os
import glob
import pickle
import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import cv2

import face_recognition
from imutils import paths
import imutils

ap= argparse.ArgumentParser()
ap.add_argument("-t", "--testfolder",
    help="Provide The folder Name of TestImages",
    default="Tarapada_Test"
)
ap.add_argument("-r", "--room",
    help="Privide The Room To compare with",
    default="FriendsRoom"
)
ap.add_argument("-s", "--score",
    help= "Enter the Evaluation Score after which the prediction are positive",
    default=50
)
args= vars(ap.parse_args())

trainedModel_path= os.path.join(
    "trainedSVM",
    args["room"]+".svm"
)

trainedLables_path= os.path.join(
    "trainedSVM",
    args["room"]+".lables"
)

testFolder_path= os.path.join(
    "..",
    "ImageData",
    "TestImages",
    args["testfolder"]
)

def markFaces(image_base, bounding_box, name, eval_score, ref_score):
    #################################################################
    #   This is a utility Function for drawing bounding boxes and
    #   putting the predicted name with score. It changes the
    #   colour if the eval_score is below ref_score.
    #   -----------------------------------------------------------
    #   return: NONE
    #
    #   Parameters:
    #       image_base: Numpy.ndArray representation of the image.
    #       bounding_box: int of detected face's location in  (top,
    #                     right, bottom, left) fromat.
    #       name: str of The Name to be printed on the predicted
    #             Lable.
    #       eval_score: float64 of the predicted score.
    #       ref_score: int/float of the reference score baced on
    #                  which the positive or negative marking will
    #                  happed.
    #################################################################
    (top, right, bottom, left)= bounding_box

    if(eval_score> ref_score):
        cv2.rectangle(
            image_base, 
            (left, top),(right, bottom),
            (0, 255, 0), 2
        )

        nameProbText_Y= top-10 if top - 10 > 10 else top + 10
        text= "{}: {:.2f}%".format(name, eval_score)
        cv2.putText(
            image_base,
            text,
            (left, nameProbText_Y),
            cv2.QT_FONT_NORMAL,
            0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
    else:
        cv2.rectangle(
            image_base, 
            (left, top),(right, bottom),
            (255, 0, 0),2
        )

testImage_paths= list(paths.list_images(testFolder_path))
if(len(testImage_paths)>0):
    for testImage_path in testImage_paths:
        testImage_name= testImage_path.split("\\")[-1]
        #testImage= face_recognition.load_image_file(testImage_path)
        #testImage= cv2.imread(testImage_path)
        testImage= imutils.resize(
            cv2.cvtColor(
                cv2.imread(testImage_path),
                cv2.COLOR_BGR2RGB
            ),
            width= 700
        )

        face_locations= face_recognition.face_locations(
            testImage,
            model="cnn",
            number_of_times_to_upsample= 1
        )

        if(len(face_locations)>0):
            face_encodings= face_recognition.face_encodings(
                testImage,
                model="cnn",
                num_jitters= 1,
                known_face_locations= face_locations
            )

            recognizerModel= pickle.loads(open(trainedModel_path, "rb").read())
            le= pickle.loads(open(trainedLables_path, "rb").read())
            predictions= recognizerModel.predict_proba(face_encodings)
            #pred_name= recognizerModel.predict(face_encodings)

            for (i, pred) in enumerate(predictions):
                highest_chans= np.argmax(pred)
                score= pred[highest_chans] * 100

                name= le.classes_[highest_chans]
                #name= le.classes_[pred_name[i]]

                markFaces(
                    testImage,
                    face_locations[i], 
                    name, score, int(args["score"])
                )

            cv2.imshow(
                testImage_name,
                cv2.cvtColor(
                    testImage,
                    cv2.COLOR_RGB2BGR
                )
            )
            cv2.waitKey(0)