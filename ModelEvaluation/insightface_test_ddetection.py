import insightface
import urllib
import urllib.request
import time
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)

#img= cv2.cvtColor(cv2.imread("ImageData\\TestImages\\Random\\177-1772654_the-it-crowd.jpg"), cv2.COLOR_BGR2RGB)

model = insightface.model_zoo.get_model('retinaface_r50_v1')

ctx_id = -1

model.prepare(ctx_id = ctx_id, nms=0.4)

start_time= time.time()
bbox, landmark = model.detect(img, threshold=0.5, scale=1.0)
print("Detected in time:", time.time()-start_time)

for bb in bbox:
  startX, startY, endX, endY, _= bb.astype(np.int)
  cv2.rectangle(img, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
cv2.imshow("", img)
cv2.waitKey(0)