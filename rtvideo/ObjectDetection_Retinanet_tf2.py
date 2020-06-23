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

from keras.models import load_model
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


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
model = models.load_model(args["model"], backbone_name='resnet50')
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

  # process image
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))

  # correct for image scale
  boxes /= scale

  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < confidence:
        break
        
      color = label_color(label)    
      b     = box.astype(int)
      name_score = str("{}:{:.3f}".format(labels_to_names[label], score))

      cv2.rectangle(output, \
                    (b[0], b[1]), (b[2], b[3]), \
                    color, \
                    2)

      cv2.rectangle(output, (b[0], b[1]), (b[0] + 120, b[1] + 25), color, thickness=-1);
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