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
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.colors import label_color
from keras.models import load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile



# get frozen graph
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


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

if not cap.isOpened():
  print("[ERROR] - Could not open video or camera")
  exit()

# load the model path

tensor_rt_graph_path = args["model"]
print(f'[INFO] - Model                = {tensor_rt_graph_path}')

print(f'[INFO] - Getting the frozen graph')

start_time = time.time()
trt_graph = get_frozen_graph(tensor_rt_graph_path)
end_time   = np.round(time.time() - start_time, 3)
print(f'[INFO] - Elapsed by {end_time} seconds')


# Create session and load graph
print(f'[INFO] Creating the session')
start_time = time.time()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')
end_time = np.round(time.time() - start_time, 3)
print(f'[INFO] - Graph completely loaded in {end_time} seconds')


input_tensor_name = 'input_1:0'
output_tensor_names = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
                       'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
                       'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0']


print(f'Gathering Tensor Names')
boxes_trt = tf_sess.graph.get_tensor_by_name(output_tensor_names[0])
scores_trt = tf_sess.graph.get_tensor_by_name(output_tensor_names[1])
labels_trt = tf_sess.graph.get_tensor_by_name(output_tensor_names[2])
predictions_trt = [boxes_trt, scores_trt, labels_trt]

labels_to_names = {0: 'fox',1: 'badger'}
confidence = float(args["conf"])

print(f'[INFO] - Starting the prediction phase')
while(cap.isOpened()):
  # capture frames
  ret, frame   = cap.read()
  if ret == True:

    output = frame.copy()
    # preprocess image for network
    frame = preprocess_image(frame)
    frame, scale = resize_image(frame, min_side=320, max_side=480)
    frame = np.expand_dims(frame, axis=0)

    # process image
    with tf_sess as sess:
        feed_dict = {input_tensor_name: frame}
        preds = sess.run(predictions_trt, feed_dict)
    boxes = preds[0]
    scores = preds[1]
    labels = preds[2]

    # correct for image scale
    boxes /= scale

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
      cv2.rectangle(output, (b[0], b[1]), (b[0] + 120, b[1] + 25), (255, 255, 0), thickness=-1);
      cv2.putText(output, name_score, (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)
      cv2.putText(output, name_score, (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Jetson NanoCam", output)
    k = cv2.waitKey(10)
    if k == 27 or k == 113 or k == 81: # ESC, q or Q
      break
  # Break the loop
  else:
    break
cap.release()
cv2.destroyAllWindows()
