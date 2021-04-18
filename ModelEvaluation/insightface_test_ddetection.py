import insightface
import urllib
import urllib.request
import time
import cv2
import numpy as np
import mxnet as mx

import imutils

from insightface.utils import face_align

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
# img = url_to_image(url)

img1= cv2.imread("C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\20191112131430_IMG_1561.JPG.jpg", cv2.COLOR_BGR2RGB)
img1= cv2.resize(img1, (700, 825))
img2= cv2.imread("C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\IMG_20190708_122350.jpg", cv2.COLOR_BGR2RGB)
img2= cv2.resize(img2, (700, 825))

img_batch= cv2.dnn.blobFromImages([img1, img2], size= (825,700))


model = insightface.model_zoo.get_model('retinaface_r50_v1')
ctx_id = -1
model.prepare(ctx_id = ctx_id, nms=0.4)

start_time= time.time()

db = mx.io.DataBatch(data=(img_batch, ))

bbox, landmark = model.detect(db, threshold=0.5, scale=1.0)

print(bbox)
exit()


for ppp in landmark:
  img2 = face_align.norm_crop(img, ppp)
  cv2.imshow("", img2)
  cv2.waitKey(0)


print("Detected in time:", time.time()-start_time)

for bb in bbox:
  startX, startY, endX, endY, _= bb.astype(np.int)
  cv2.rectangle(img, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
cv2.imshow("", img)
cv2.waitKey(0)