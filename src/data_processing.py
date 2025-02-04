import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import xml.etree.ElementTree as ET
import time

class DataSplit:
    """
    A class to handle the splitting of a dataset into training, validation, and test sets.
    It also supports converting annotations to YOLO format and organizing the dataset accordingly.

    Attributes:
        dataset_type (str): The type of dataset to process. Supported types are 'PL2K' and 'LogoDet-3K'.
        class_map (dict): A dictionary to map class names to unique integer IDs.
    """

    def __init__(self, dataset_type='PL2K'):
        """
        Initializes the DataSplit class.

        Args:
            dataset_type (str, optional): The type of dataset to process. Defaults to 'PL2K'.
        """
        self.dataset_type = dataset_type
        self.class_map = {}

    def load_data(self, input_path, output_path, batch_size=32):
        """
        Loads the dataset from the input path, splits it into training, validation, and test sets,
        and saves the split datasets to the output path.

        Args:
            input_path (str): The path to the input dataset.
            output_path (str): The path to save the split datasets.
            batch_size (int, optional): The batch size for data loading. Defaults to 32.
        """
        # Create directories
        splits = ['train', 'val', 'test']
        for split in splits:
            os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'annotations', split), exist_ok=True)

        # Load and split data
        annotations, images = self._get_annotations(input_path)
        train_img, test_img, train_ann, test_ann = train_test_split(
            images, annotations, test_size=0.2, random_state=42
        )
        val_img, test_img, val_ann, test_ann = train_test_split(
            test_img, test_ann, test_size=0.5, random_state=42
        )

        # Copy files and convert annotations to YOLO format
        self._copy_files(train_img, train_ann, output_path, 'train')
        self._copy_files(val_img, val_ann, output_path, 'val')
        self._copy_files(test_img, test_ann, output_path, 'test')

    def _copy_files(self, images, annotations, output_path, split):
        """
        Copies image and annotation files to the specified output directory and converts annotations to YOLO format.

        Args:
            images (list): List of image file paths.
            annotations (list): List of annotation file paths.
            output_path (str): The path to save the copied files.
            split (str): The dataset split ('train', 'val', or 'test').
        """
        for img, ann in zip(images, annotations):
            try:
                # Generate unique filenames
                unique_id = str(int(time.time() * 1000))  # Use current timestamp as unique identifier
                img_filename = f"{os.path.splitext(os.path.basename(img))[0]}_{unique_id}.jpg"
                ann_filename = f"{os.path.splitext(os.path.basename(ann))[0]}_{unique_id}.xml"
                
    
                # Copy image
                shutil.copy(img, os.path.join(output_path, 'images', split, img_filename))
                shutil.copy(ann, os.path.join(output_path, 'annotations', split, ann_filename))

            except Exception as e:
                print(f"Error copying {img}: {e}")

    def _get_annotations(self, path):
        """
        Retrieves the annotations and corresponding image paths based on the dataset type.

        Args:
            path (str): The path to the dataset.

        Returns:
            tuple: A tuple containing two lists: annotations and images.

        Raises:
            ValueError: If the dataset type is unsupported.
        """
        if self.dataset_type == 'PL2K':
            return self._get_pl2k_annotations(path)
        elif self.dataset_type == 'LogoDet-3K':
            return self._get_logodet_annotations(path)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _get_pl2k_annotations(self, path):
        """
        Retrieves annotations and image paths for the PL2K dataset.

        Args:
            path (str): The path to the PL2K dataset.

        Returns:
            tuple: A tuple containing two lists: annotations and images.
        """
        annotations = []
        images = []
        for brand_dir in glob(os.path.join(path, '*')):
            for img_path in glob(os.path.join(brand_dir, '*.jpg')):
                xml_path = img_path.replace('.jpg', '.xml')
                if os.path.exists(xml_path):
                    images.append(img_path)
                    annotations.append(xml_path)
        return annotations, images

    def _get_logodet_annotations(self, path):
        """
        Retrieves annotations and image paths for the LogoDet-3K dataset.

        Args:
            path (str): The path to the LogoDet-3K dataset.

        Returns:
            tuple: A tuple containing two lists: annotations and images.
        """
        annotations = []
        images = []
        for folder in glob(os.path.join(path, '*'), recursive=True):
            for subfolder in glob(os.path.join(folder, '*'), recursive=True):
                for txt in glob(os.path.join(subfolder, '*.xml')):
                    annotations.append(txt)
                    jpg = txt.replace('.xml', '.jpg')
                    images.append(jpg)
        return annotations, images