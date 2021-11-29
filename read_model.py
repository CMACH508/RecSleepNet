import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = "/home/niehaodong/tsn/out_sleepedf_nobn/train/19/best_ckpt/best_model.ckpt-27236.index"
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
	print("tensor_name: ", key)
	print(reader.get_tensor(key))
