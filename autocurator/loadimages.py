import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy

def build_data(x_files, y_files):

  to_del = 0
  cnt1 = -1

  for k in range(len(y_files)):
    cnt1 = cnt1 + 1
    tmp1 = scipy.io.loadmat(x_files[cnt1])
    tmp2 = scipy.io.loadmat(y_files[cnt1])

    Xtmp = tmp1['finalMat']
    Ytmp = tmp2['touchClass']
    if cnt1==0:
      raw_X = Xtmp
      raw_Y = Ytmp
    else:

      raw_X = np.concatenate((raw_X,Xtmp), axis=0)
      raw_Y = np.concatenate((raw_Y,Ytmp), axis=0)

  return raw_X, raw_Y

class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, file_trial_list, file_Y_list, num_in_each, batch_size):
        cnt = 0
        extract_inds = []
        # num_in_each contains the number of frames in each file I am loading, ie
        # for trial/file 1 there are 200 frames , trial/file 2 has 215 frames etc
        for k, elem in enumerate(num_in_each):
            tot_frame_nums = sum(num_in_each[cnt: k + 1])  # used to test if the number of frames in
            # all these files exceded the "batch_size" limit
            if tot_frame_nums > batch_size or len(num_in_each) - 1 == k:  # condition met, these files together
                # meet the max requirment to load together as a batch
                extract_inds.append([cnt, k + 1])
                cnt = k + 1  # reset to the current iter
                if np.diff(
                        extract_inds[-1]) > 1:  # if there is more than one file then we want to take off the last file
                    # because it excedes the set number of frames
                    extract_inds[-1][-1] = extract_inds[-1][-1] - 1
                    cnt = cnt - 1

        file_list_chunks = []
        file_Y_list_chunks = []
        for i, ii in enumerate(extract_inds):
            file_list_chunks.append(file_trial_list[ii[0]:ii[1]])
            file_Y_list_chunks.append(file_Y_list[ii[0]:ii[1]])

        self.file_trial_list = file_trial_list
        self.file_Y_list = file_Y_list
        self.batch_size = batch_size
        self.extract_inds = extract_inds
        self.num_in_each = num_in_each
        self.file_list_chunks = file_list_chunks
        self.file_Y_list_chunks = file_Y_list_chunks

    def __len__(self):
        return len(self.extract_inds)

    def __getitem__(self, num_2_extract):
        raw_X, raw_Y = build_data(self.file_list_chunks[num_2_extract],
                                  self.file_Y_list_chunks[num_2_extract])

        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        IMG_SIZE = 96  # All images will be resized to 96x96. This is the size of MobileNetV2 input sizes

        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (IMG_SIZE, IMG_SIZE))  # resizing

        self.IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        rgb_tensor_aug = rgb_tensor
        return rgb_tensor_aug, raw_Y

    def get_single_trials(self, num_2_extract):
        raw_X, raw_Y = build_data([self.file_trial_list[num_2_extract]],
                                  [self.file_Y_list[num_2_extract]])

        frame_index = scipy.io.loadmat(self.frame_ind_list[num_2_extract])
        frame_index = frame_index['relevantIdx']
        frame_index = frame_index[0]
        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        IMG_SIZE = 96  # All images will be resized to 160x160. This is the size of MobileNetV2 input sizes

        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (IMG_SIZE, IMG_SIZE))  # resizing

        self.IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        rgb_tensor_aug = rgb_tensor

        return rgb_tensor_aug, raw_Y

        # return rgb_tensor, raw_Y, frame_index#, trial_file_num


    def plot_batch_distribution(self):
        # randomly select a batch and generate images and labels
        batch_num = np.random.choice(np.arange(0, len(self.file_list_chunks)))
        samp_x, samp_y = self.__getitem__(batch_num)

        # look at the distribution of classes
        plt.pie([1 - np.mean(samp_y), np.mean(samp_y)],
                labels=['non-touch frames', 'touch frames'], autopct='%1.1f%%', )
        plt.title('class distribution from batch ' + str(batch_num))
        plt.show()

        # generate indices for positive and negative classes
        images_to_sample = 20
        neg_class = [i for i, val in enumerate(samp_y) if val == 0]
        pos_class = [i for i, val in enumerate(samp_y) if val == 1]
        neg_index = np.random.choice(neg_class, images_to_sample)
        pos_index = np.random.choice(pos_class, images_to_sample)

        # plot sample positive and negative class images
        plt.figure(figsize=(10, 10))
        for i in range(images_to_sample):
            plt.subplot(5, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            _ = plt.imshow(image_transform(samp_x[neg_index[i]]))
            plt.xlabel('0')

            plt.subplot(5, 10, images_to_sample + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image_transform(samp_x[pos_index[i]]))
            plt.xlabel('1')
        plt.suptitle('sample images from batch  ' + str(batch_num))
        plt.show()