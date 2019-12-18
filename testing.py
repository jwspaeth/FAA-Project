#!/usr/bin/env python3

from lib.classes.dataset_classes.FaaDataset import FaaDataset

def main():
	test_dataset = FaaDataset()
	test_dataset._load_raw_data()
	training_inds, val_inds, feature_array, label_array = test_dataset._load_data()

if __name__ == "__main__":
	main()