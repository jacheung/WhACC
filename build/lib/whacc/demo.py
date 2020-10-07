import whacc
import os

# set directories for dataset
h5_file_directory = os.getcwd() + '/whacc/model_checkpoints/'

# get h5 file information
h5_file_list, total_image_count = whacc.get_h5_info(h5_file_directory)

# build image batch generator
batch_size = 2000
batch_generator = whacc.ImageBatchGenerator(batch_size, h5_file_list, total_image_count)

# load model using existing checkpoints
model = whacc.load_model()

# predict images
predictions = model.predict(batch_generator)

