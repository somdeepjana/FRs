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

batch_size= 64


def markFaces(image_base, bounding_box, name, eval_score, ref_score):
    """This is a utility function for drawing bounding boxes.
    
    It put the predicted name with score and it changes the colour if the eval_score is below ref_score.

    Args:
        image_base (Numpy.ndArray): The Bace Image on which the marking will happen
        bounding_box (int tuple): Detected face's location in  (top, right, bottom, left) fromat.
        name (str): The Name to be printed on the predicted Lable.
        eval_score (float64): The predicted score.
        ref_score (int/float): The reference score baced on which the positive or negative marking will happed.

    Returns:
        BOOLEAN: True when eval_score> ref_score, False otherwise, It also determines whether the rectrangel is drawn.
    """
    (top, right, bottom, left)= bounding_box
    
    if(eval_score> ref_score and name != "unknown"):
        cv2.rectangle(
            image_base, 
            (left, top),(right, bottom),
            (0, 255, 0), 2
        )

        nameProbText_Y= top-10 if top - 10 > 10 else top + 10
        text= "{}: {:.2f}".format(name, eval_score)
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
        nameProbText_Y= top-10 if top - 10 > 10 else top + 10
        text= "{}: {:.2f}".format(name, eval_score)
        cv2.putText(
            image_base,
            text,
            (left, nameProbText_Y),
            cv2.QT_FONT_NORMAL,
            0.5, (255, 0, 0), 1, cv2.LINE_AA
        )
        return False