import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import random
random.seed(42)

class FewShotLogoDataset(Dataset):
    """
    A PyTorch Dataset class for few-shot logo detection tasks. It supports creating support and query sets
    for few-shot learning by grouping images by brand and splitting them into support and query sets.

    Args:
        images_dir (str): Path to the directory containing images (e.g., "data/process/images/val").
        annotations_dir (str): Path to the directory containing XML annotations (e.g., "data/process/annotation/val").
        support_samples (int, optional): Number of support samples per class. Defaults to 5.
    """

    def __init__(self, images_dir: str, annotations_dir: str, support_samples: int = 5):
        """
        Initializes the FewShotLogoDataset class.

        Args:
            images_dir (str): Path to the directory containing images.
            annotations_dir (str): Path to the directory containing XML annotations.
            support_samples (int, optional): Number of support samples per class. Defaults to 5.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.support_samples = support_samples
        
        # Collect data and labels
        self.image_paths, self.brand_labels = self._parse_annotations()
        
        # Group images by brand
        self.brand_groups = defaultdict(list)
        for img_path, brand in zip(self.image_paths, self.brand_labels):
            self.brand_groups[brand].append(img_path)
            
        # Create support and query sets
        self.support_set, self.query_set = self._split_support_query()

    def _parse_annotations(self):
        """
        Parses XML annotations to collect image paths and corresponding brand labels.

        Returns:
            tuple: A tuple containing two lists: image paths and brand labels.
        """
        image_paths = []
        brand_labels = []
        
        # Collect all images
        for img_file in os.listdir(self.images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.images_dir, img_file)
                xml_path = os.path.join(
                    self.annotations_dir, 
                    os.path.splitext(img_file)[0] + ".xml"
                )
                
                # Parse XML
                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        
                        # Find the first object with a brand name
                        for obj in root.findall('object'):
                            brand = obj.find('name').text
                            if brand:
                                image_paths.append(img_path)
                                brand_labels.append(brand)
                                break  # Take the first found brand
                    except Exception as e:
                        print(f"Error parsing {xml_path}: {str(e)}")
                        
        return image_paths, brand_labels

    def _split_support_query(self):
        """
        Splits the dataset into support and query sets based on the number of support samples per class.

        Returns:
            tuple: A tuple containing two lists: support set and query set.
        """
        support = []
        query = []
        
        for brand, paths in self.brand_groups.items():
       
            support.extend([(p, brand) for p in paths])
            query.extend([(p, brand) for p in paths])
            
        return support, query

    def get_support_set(self):
        """
        Returns the support set.

        Returns:
            list: A list of tuples containing image paths and brand labels for the support set.
        """
        return self.support_set

    def get_query_set(self):
        """
        Returns the query set.

        Returns:
            list: A list of tuples containing image paths and brand labels for the query set.
        """
        return self.query_set

    def __len__(self):

        return len(self.query_set)
    
    def __getitem__(self, idx):
    
        img_path, brand = self.query_set[idx]
        image = Image.open(img_path).convert('RGB')
        return image, brand