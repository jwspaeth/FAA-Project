
import math

def get_size_input(conv_config, max_pool_config, total_steps, total_seconds):
	
	conv_output_size = math.ceil(total_steps / conv_config.n_strides)
	size_input = math.ceil(conv_output_size * (max_pool_config.size_in_seconds / total_seconds))

	return size_input