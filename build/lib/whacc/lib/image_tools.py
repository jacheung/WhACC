import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob


def get_h5_info(h5_file_directory):
    h5_file_list = glob.glob(h5_file_directory + "*.h5")
    total_frame_count = []
    for H5_file in h5_file_list:
        H5 = h5py.File(H5_file, 'r')
        images = H5['images']
        total_frame_count.append(images.shape[0])

    return h5_file_list, total_frame_count

def batch_size_file_ind_selector(num_in_each, batch_size):
    """batch_size_file_ind_selector - needed for ImageBatchGenerator to know which H5 file index
    to use depending on the iteration number used in __getitem__ in the generator.
    this all depends on the variable batch size.

    Example: the output of the following...
    batch_size_file_ind_selector([4000, 4001, 3999], [2000])
    would be [0, 0, 1, 1, 1, 2, 2] which means that there are 2 chunks in the first
    H5 file, 3 in the second and 2 in the third based on chunk size of 2000 """
    break_into = np.ceil(np.array(num_in_each)/batch_size)
    extract_inds = np.array([])
    for k, elem in enumerate(break_into):
        tmp1 = np.array(np.ones(np.int(elem))*k)
        extract_inds = np.concatenate((extract_inds, tmp1), axis=0)
    return extract_inds


# file_inds_for_H5_extraction is the same as extract_inds output from the above function
def reset_to_first_frame_for_each_file_ind(file_inds_for_H5_extraction):
    """reset_to_first_frame_for_each_file_ind - uses the output of batch_size_file_ind_selector
    to determine when to reset the index for each individual H5 file. using the above example
    the out put would be [0, 0, 2, 2, 2, 5, 5], each would be subtracted from the indexing to
    set the position of the index to 0 for each new H5 file."""
    subtract_for_index = []
    for k, elem in enumerate(file_inds_for_H5_extraction):
        tmp1 = np.diff(file_inds_for_H5_extraction)
        tmp1 = np.where(tmp1!=0)
        tmp1 = np.append(-1, tmp1[0])+1
        subtract_for_index.append(tmp1[np.int(file_inds_for_H5_extraction[k])])
    return subtract_for_index


class ImageBatchGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, h5_file_list,num_frames_in_all_H5_files):
        file_inds_for_H5_extraction = batch_size_file_ind_selector(
            num_frames_in_all_H5_files, batch_size)
        subtract_for_index = reset_to_first_frame_for_each_file_ind(
            file_inds_for_H5_extraction)
        # self.to_fit = to_fit #set to True to return XY and False to return X
        self.batch_size = batch_size
        self.H5_file_list = h5_file_list
        self.num_frames_in_all_H5_files = num_frames_in_all_H5_files
        self.file_inds_for_H5_extraction = file_inds_for_H5_extraction
        self.subtract_for_index = subtract_for_index
        self.IMG_SIZE = 96

    def __len__(self):
        return len(self.file_inds_for_H5_extraction)

    def __getitem__(self, num_2_extract):
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        H5 = h5py.File(H5_file, 'r')
        #  list(H5.keys())

        images = H5['images']
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        rgb_tensor = self.image_transform(raw_X)

        # if self.to_fit:
        #   labels_tmp = H5['labels']
        #   raw_Y = labels_tmp[b*num_2_extract_mod:b*(num_2_extract_mod+1)]
        #   return rgb_tensor, raw_Y
        # else:
        return rgb_tensor

    def getXandY(self, num_2_extract):
        b = self.batch_size
        h = self.H5_file_list
        i = self.file_inds_for_H5_extraction
        H5_file = h[np.int(i[num_2_extract])]
        H5 = h5py.File(H5_file, 'r')
        #  list(H5.keys())

        images = H5['images']
        num_2_extract_mod = num_2_extract - self.subtract_for_index[num_2_extract]
        raw_X = images[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        rgb_tensor = self.image_transform(raw_X)
        labels_tmp = H5['labels']
        raw_Y = labels_tmp[b * num_2_extract_mod:b * (num_2_extract_mod + 1)]
        return rgb_tensor, raw_Y

    def image_transform(self, raw_X):
        # input num_of_images x H x W, image input must be grayscale
        # MobileNetV2 requires certain image dimensions
        # We use N x 61 x 61 formated images
        # self.IMG_SIZE is a single number to change the images into, images must be square
        rgb_batch = np.repeat(raw_X[..., np.newaxis], 3, -1)
        rgb_tensor = tf.cast(rgb_batch, tf.float32)  # convert to tf tensor with float32 dtypes
        rgb_tensor = (rgb_tensor / 127.5) - 1  # /127.5 = 0:2, -1 = -1:1 requirement for mobilenetV2
        rgb_tensor = tf.image.resize(rgb_tensor, (self.IMG_SIZE, self.IMG_SIZE))  # resizing
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
        return rgb_tensor

    def plot_batch_distribution(self):
        # randomly select a batch and generate images and labels
        batch_num = np.random.choice(np.arange(0, self.__len__()))
        samp_x, samp_y = self.getXandY(batch_num)

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
        samp_x = (samp_x + 1) / 2
        for i in range(images_to_sample):
            plt.subplot(5, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            _ = plt.imshow(samp_x[neg_index[i]])
            plt.xlabel('0')

            plt.subplot(5, 10, images_to_sample + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(samp_x[pos_index[i]])
            plt.xlabel('1')
        plt.suptitle('sample images from batch  ' + str(batch_num))
        plt.show()