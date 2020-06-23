# Example of usage below
# python ObjectDetection_Retinanet.py -v <video_path> -c <confidence_threshold> -m <model_path>

# if current working is = / and has the file ObjectDetection_Retinanet.py
# if videos and model are on /home/nvidia/deeplearning
#   but in separate subfolders 
#     the model -> /home/nvidia/deeplearning/model
#     the video -> /home/nvidia/deeplearning/video
#
# python ObjectDetection_Retinanet.py \
# -v /home/nvidia/deeplearning/video/video1.mp4 \
# -c 0.5 \
# -m /home/nvidia/deeplearning/model/output.h5

#import neccesary modules
import os
import cv2
import argparse
import numpy as np

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras.models import load_model


# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", \
	            required=True,   \
	            help="URL of video")

ap.add_argument("-c", "--conf", \
	            required=True,  \
	            help="Confidence threshold")

ap.add_argument("-m", "--model", \
	            required=True,  \
	            help="Path of Deep Learning Model")

args = vars(ap.parse_args())


print(f'[INFO] - Processing video     = {args["video"]}')
print(f'[INFO] - Confidence Threshold = {args["conf"]}')

if not os.path.exists(args["video"]):
	print("[ERROR] - Please Check path of the video")
	exit()

if not os.path.exists(args["model"]):
  print("[ERROR] - Please Check path of the model")
  exit()

cap = cv2.VideoCapture(args["video"])

# Check if file exists
if (cap.isOpened()== False): 
  print("[ERROR] - Could not open video or camera")
  exit()

# load the model
model = load_model(args["model"], custom_objects=custom_objects)

labels_to_names = {0: 'fox',   \
                   1: 'badger' \
                  }
confidence = float(args["conf"])

while(cap.isOpened()):
  # capture frames
  ret, frame   = cap.read()

  # copy to draw on
  output = frame.copy()

  # preprocess image for network
  frame = preprocess_image(frame)
  frame, scale = resize_image(frame)#, min_side=320, max_side=480)

  # detect objects in the input image and correct for the image scale
  boxes, nmsClassification = model.predict_on_batch(image)[2:2] 
  boxes /= scale

  # compute the predicted labels and probabilities
  predLabels = np.argmax(nmsClassification[0, :, :], axis=1)
  scores = nmsClassification[0,
  np.arange(0, nmsClassification.shape[1]), predLabels]

  # loop over the detections
  for (i, (label, score)) in enumerate(zip(predLabels, scores)):
  # filter out weak detections
      if score < args["confidence"]:
          continue

      # grab the bounding box for the detection
      b = boxes[0, i, :].astype("int")

      # build the label and draw the label + bounding box on the output
      # image
      label = "{}: {:.2f}".format(LABELS[label], score)

      cv2.rectangle(output, \
                    (b[0], b[1]), (b[2], b[3]), \
                    color, \
                    2)

      cv2.rectangle(output, (b[0], b[1]), (b[0] + 120, b[1] + 25), (255, 255, 0), thickness=-1);
      cv2.putText(output, name_score, (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)
      cv2.putText(output, name_score, (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)


  if ret == True:
    #output = cv2.resize(output, (320, 240))
    cv2.imshow("Jetson NanoCam", output)    
    k = cv2.waitKey(10)
    if k == 27 or k == 113 or k == 81: # ESC, q or Q
      break
  # Break the loop
  else: 
    break
cap.release()
cv2.destroyAllWindows()