############################################################################
#
#   For Configuration Settings and Utility fuctions
#
############################################################################
import os

import cv2


# thisFile_path= os.path.dirname(os.path.abspath(__file__))
# print(thisFile_path)
imagesToRegister_path= os.path.join(
    # thisFile_path,
    "..",
    "ImageData", 
    "ImagesToRegister"
)

registerData_path= "RegisteredData"

trainedModdels_path= "trainedSVM"


def markFaces(image_base, bounding_box, name, eval_score, ref_score):
    #################################################################
    #   This is a utility Function for drawing bounding boxes and
    #   putting the predicted name with score. It changes the
    #   colour if the eval_score is below ref_score.
    #   -----------------------------------------------------------
    #   return: BOOLEAN
    #           True when eval_score> ref_score, False otherwise, It
    #           also determines whether the rectrangel is drawn.
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
        return True
    else:
        cv2.rectangle(
            image_base, 
            (left, top),(right, bottom),
            (255, 0, 0),2
        )
        return False