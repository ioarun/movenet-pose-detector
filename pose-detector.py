import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import glob

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches


from helpers import draw_prediction_on_image


model_name = "movenet_lightning"

if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoint_with_scores = outputs['output_0'].numpy()
    return keypoint_with_scores

# Load the input image.
# One of the URFD data folders
DIR_PATH = "sample-data/sample-elderly-falling"
TARGET_PATH = "sample-output/sample-output-elderly"
# Sample the first image
IMAGE_PATHS = sorted(glob.glob(DIR_PATH+"/*"))

for IMAGE_PATH in IMAGE_PATHS:
    image = tf.io.read_file(IMAGE_PATH)
    image = tf.image.decode_jpeg(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoint_with_scores = movenet(input_image)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoint_with_scores)

    im_bgr = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)


    # print (keypoint_with_scores, keypoint_with_scores.shape)

    file_name = IMAGE_PATH.split("/")[2]
    print (TARGET_PATH+"/"+file_name)
    cv2.imwrite(TARGET_PATH+"/"+file_name, im_bgr)

# plt.figure(figsize=(16, 8))
# plt.imshow(output_overlay)
# _ = plt.axis('off')
# plt.show()
