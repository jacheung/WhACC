import os
import glob
import cv2
import numpy as np
import time
import re
import h5py
import matplotlib.pyplot as plt


class PoleTracking():
    def __init__(self, video_directory, template_image):
        self.video_directory = video_directory
        self.video_files = glob.glob(os.path.join(video_directory, '*.mp4'))
        self.template_image = template_image

    @staticmethod
    def crop_image(image, crop_values, inflation=1):
        """
        This is an accessory function to track to improve tracking speed. This crops the initial large image into a smaller one, based on the inflation rate.
        Inflation rate of 1 = 3 x 3 template image size around the first guessed pole location.
        """
        crop_top_left = crop_values['top_left'] - np.array(crop_values['template_shape'] * inflation)
        crop_bottom_right = crop_values['bottom_right'] + np.array(crop_values['template_shape'] * inflation)

        # You cannot crop larger than the size of the image. Defaults to max value. So no need to worry about inflation being too high.
        cropped_image = image[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]

        return cropped_image

    def track(self, video_file, match_method = 'cv2.TM_CCOEFF'):
        """
        this function scans a template image across each frame of the video to identify the pole location.
        This assumes there is a pole at each frame. Cropping optimizes scanning by ~80% and uses the first frame
        as a point of reference.
        """

        # width and height of img_stacks will be that of template (61x61)
        w, h = self.template_image.shape[::-1]

        # open video at directory
        video = cv2.VideoCapture(video_file)
        if (video.isOpened() == False):
          print('error opening video file')

        fno = 0
        img_list = []
        loc_list = []
        success, frame = video.read()
        method = eval(match_method)

        while success:
          # preprocess image
          if 'crop_values' in locals():
            frame = self.crop_image(frame, crop_values)
          img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          img_raw = img_raw.astype(np.float32)
          img = img_raw.copy()

          # Apply template Matching
          res = cv2.matchTemplate(img,self.template_image,method)
          min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

          # get coordinates of matched target
          top_left = max_loc
          bottom_right = (top_left[0] + w, top_left[1] + h)

          if 'crop_values' in locals():
            # readjusting location to main image
            max_loc = max_loc + np.array(crop_values['top_left']) - np.array(crop_values['template_shape'])
          else:
            crop_values = {'top_left': top_left, 'bottom_right': bottom_right, 'template_shape': self.template_image.shape}

          # crop image and store
          crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
          img_list.append(crop_img)
          loc_list.append(max_loc)

          # iterate to next frame and crop using current details
          fno += 1
          success, frame = video.read()

        img_stack = np.array(img_list, dtype=np.uint8)
        loc_stack = np.array(loc_list)
        return img_stack, loc_stack

    def plot_pole_center(self, video_file, location_stack):
        video = cv2.VideoCapture(video_file)
        success, frame = video.read()
        location_stack = location_stack + np.array(self.template_image.shape)/2

        plt.imshow(frame)
        plt.scatter(location_stack[:,0], location_stack[:,1], s=.2, c='r')
        plt.show()

        threshold = 50   # threshold in pixels
        distance_from_mean = np.sum(np.abs(location_stack - np.mean(location_stack,axis=0)),axis=1)
        exceed_threshold = np.mean(distance_from_mean < threshold) * 100
        print(str(np.round(exceed_threshold,2)) + '% of the pole locations are within ' + str(threshold) + ' pixels from the mean pole location.')

    def track_random(self):
        """
        This function will randomly grab a video file within the directory and
        track all frames within there and plot tracked locations over the original image
        """
        video_file = np.random.choice(self.video_files, 1)[0]
        print('Testing tracking on ' + video_file)
        img_stack, loc_stack = self.track(video_file = video_file)
        self.plot_pole_center(video_file = video_file, location_stack = loc_stack)

    def track_all_and_save(self):
        """
        This is the major output function of the PoleTracking class. This functions will
        track all video files within the directory and save an H5 file with the meta information:
        - file_name_nums : this is the trial number extracted from the video file name
        - image: the tracked image stack
        - labels : this is pre-allocation for the CNN to label the images
        - trial_nums_and_frame_nums: 2 dimensional vector. Row 1 = trial number and Row 2 = frame number in that trial
        - in_range: this is pre-allocation for the pole in range tracker
        """
        # create image stack
        final_stack = []
        start = time.time()
        for video in self.video_files:
          print('Tracking... ' + video)
          img_stack, loc_stack = self.track(video_file = video)
          final_stack.extend(img_stack)
        elapsed = time.time() - start
        print('Tracker runtime : ' + str(elapsed/60) + ' mins')

        # pull trial numbers and frame numbers
        trial_nums = list(map(lambda s: re.search("^.*-([0-9]+)\.", s).group(1), self.video_files))
        frame_nums = list(map(lambda s: cv2.VideoCapture(s).get(7), self.video_files))

        tnf = np.vstack([np.array(list(trial_nums)).astype(int),
                  np.array(list(frame_nums))])

        # populating which trial each frame is in
        fnn = []
        for a,b in zip(list(trial_nums),list(frame_nums)):
          fnn = np.concatenate([fnn,np.repeat(int(a),b)])

        # populating "labels" with -1
        labels = np.ones(img_stack.shape[0]) * -1

        # populating whether pole "in_range" with nan values
        in_range = np.empty(labels.shape)
        in_range[:] = np.nan

        # check to make sure sizes across file names and images are equal
        assert(len(fnn) == len(final_stack))

        # save data in H5 with name similar to video
        file_name = re.search("([^\/]+)\-",self.video_files[0])[0]
        path_with_name = file_name + '.h5'
        print('H5 file saving under the name ' + path_with_name)
        print('and placed in ' + self.video_directory)
        hf = h5py.File(self.video_directory + file_name + '.h5', 'w')

        hf.create_dataset('file_name_nums', data=fnn)
        hf.create_dataset('images', data=final_stack)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('trial_nums_and_frame_nums', data = tnf)
        hf.create_dataset('in_range', data = in_range)

        hf.close()
        return h5py.File(self.video_directory + file_name + '.h5','r')


