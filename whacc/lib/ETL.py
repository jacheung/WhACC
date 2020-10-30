import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import glob


def pin_tracker(video_directory, template, match_method):
    """ function to find template image for each frame using template matching in OpenCV """
    # open video at directory
    video = cv2.VideoCapture(video_directory)
    if (video.isOpened() == False):
        print('error opening video file')


    w, h = template.shape[::-1]
    fno = 0
    img_list = []
    loc_list = []
    success, frame = video.read()
    method = eval(match_method)

    while success:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # get coordinates of matched target
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # crop image and store
        crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        img_list.append(crop_img)
        loc_list.append(max_loc)
        # iterate to next frame
        fno += 1
        success, frame = video.read()

    img_stack = np.array(img_list)
    loc_stack = np.array(loc_list)
    return img_stack, loc_stack

def plot_pole_center(video_directory, template_img, location_stack):
  video = cv2.VideoCapture(video_directory)
  success, frame = video.read()
  plt.imshow(frame)
  w = template_img.shape[0] / 2
  h = template_img.shape[1] / 2
  for k in location_stack:
    plt.scatter(k[0] + w,k[1]+h, s=.2, c='r')
  plt.show()

  threshold = 50
  distance_from_mean = np.sum(np.abs(location_stack - np.mean(location_stack,axis=0)),axis=1)
  exceed_threshold = np.mean(distance_from_mean < threshold) * 100
  print(str(np.round(exceed_threshold,2)) + '% of the pole locations are within ' + str(threshold) + ' pixels from the mean pole location.')
