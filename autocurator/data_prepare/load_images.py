import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy


class ImageBatchGenerator(keras.utils.Sequence):

    def __init__(self, file_trial_list, file_Y_list, num_in_each, batch_size, to_fit):
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

        self.to_fit = to_fit  # set to True to return XY and False to return X
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

        raw_X = self._generate_X(self.file_list_chunks[num_2_extract])

        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        IMG_SIZE = 96  # All images will be resized to 160x160. This is the size of MobileNetV2 input sizes

        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (IMG_SIZE, IMG_SIZE))  # resizing

        self.IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        rgb_tensor_aug = rgb_tensor

        # print(len(raw_Y))
        # for i, ims in enumerate(rgb_tensor):
        #   # print(i)
        #   tmp1 = fux_wit_imgs(20, ims)
        #   rgb_tensor_aug = np.append(rgb_tensor_aug, tmp1, 0)

        if self.to_fit:
            raw_Y = self._generate_Y(self.file_Y_list_chunks[num_2_extract])
            return rgb_tensor_aug, raw_Y
        else:
            return rgb_tensor_aug

    def get_single_trials(self, num_2_extract):

        raw_X = self._generate_X(self.file_list_chunks[num_2_extract])
        raw_Y = self._generate_Y(self.file_Y_list_chunks[num_2_extract])

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
        # print(len(raw_Y))
        # for i, ims in enumerate(rgb_tensor):
        #   print(i)
        #   tmp1 = fux_wit_imgs(20, ims)
        #   rgb_tensor_aug = np.append(rgb_tensor_aug, tmp1, 0)
        return rgb_tensor_aug, raw_Y

        # return rgb_tensor, raw_Y, frame_index#, trial_file_num

    # Function to generate an image tensor and corresponding label array

    def _build_data(self, x_files, y_files):
        """Phils original build data structure used to generate X and Y together. It has been broken down into _generate_X and _generate_Y. Delete ASAP"""
        cnt1 = -1;

        for k in range(len(y_files)):
            cnt1 = cnt1 + 1
            tmp1 = scipy.io.loadmat(x_files[cnt1])
            tmp2 = scipy.io.loadmat(y_files[cnt1])

            Xtmp = tmp1['finalMat']
            Ytmp = tmp2['touchClass']
            if cnt1 == 0:
                raw_X = Xtmp
                raw_Y = Ytmp
            else:

                raw_X = np.concatenate((raw_X, Xtmp), axis=0)
                raw_Y = np.concatenate((raw_Y, Ytmp), axis=0)

        return raw_X, raw_Y

    def _generate_X(self, x_files):
        cnt1 = -1;

        for k in range(len(x_files)):
            cnt1 = cnt1 + 1
            tmp1 = scipy.io.loadmat(x_files[cnt1])
            Xtmp = tmp1['finalMat']

            if cnt1 == 0:
                raw_X = Xtmp
            else:
                raw_X = np.concatenate((raw_X, Xtmp), axis=0)

        return raw_X

    def _generate_Y(self, y_files):
        cnt1 = -1;

        for k in range(len(y_files)):
            cnt1 = cnt1 + 1
            tmp2 = scipy.io.loadmat(y_files[cnt1])

            Ytmp = tmp2['touchClass']
            if cnt1 == 0:
                raw_Y = Ytmp
            else:
                raw_Y = np.concatenate((raw_Y, Ytmp), axis=0)

        return raw_Y

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