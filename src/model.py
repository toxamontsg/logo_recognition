from typing import Dict, List, Tuple, Optional
import torch
import clip
from PIL import Image
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from ultralytics import YOLO

class FewShotLogoRecognizer:
    """
    A class for few-shot logo recognition using CLIP and YOLO.

    Attributes:
        device (str): The device to run the model on (either 'cuda' or 'cpu').
        model (torch.nn.Module): The CLIP model used for image embeddings.
        preprocess (callable): The preprocessing function for CLIP.
        brand_prototypes (Dict[str, torch.Tensor]): A dictionary mapping brand names to their prototype embeddings.
        detector (YOLO): The YOLO object detector, if a detector path is provided.
        detector_conf (float): Confidence threshold for YOLO detections.
        detector_iou (float): IoU threshold for YOLO detections.
    """

    def __init__(self, brand_examples: Dict[str, List[str]], model_name: str = "ViT-B/32", detector_path: str = None):
        """
        Initializes the FewShotLogoRecognizer.

        Args:
            brand_examples (Dict[str, List[str]]): A dictionary mapping brand names to lists of image paths.
            model_name (str): The name of the CLIP model to use. Default is "ViT-B/32".
            detector_path (str): The path to the YOLO detector model, if any.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.brand_prototypes = self._create_prototypes(brand_examples)

        # Initialize the detector
        self.detector = YOLO(detector_path) if detector_path else None
        self.detector_conf = 0.3
        self.detector_iou = 0.5

    def _create_prototypes(self, brand_examples: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Creates prototype embeddings for each brand.

        Args:
            brand_examples (Dict[str, List[str]]): A dictionary mapping brand names to lists of image paths.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping brand names to their prototype embeddings.
        """
        prototypes = {}
        for brand, paths in brand_examples.items():
            embeddings = []
            for path in paths:
                try:
                    image = self.preprocess(Image.open(path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        emb = self.model.encode_image(image)
                        emb /= emb.norm(dim=-1, keepdim=True)
                        embeddings.append(emb)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            if embeddings:
                avg_embedding = torch.mean(torch.cat(embeddings), dim=0)
                avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
                prototypes[brand] = avg_embedding
        return prototypes

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesses an image for the CLIP model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        img = cv2.resize(image, (608, 608))
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    def detect_logos(self, image_path: str, target_brand: Optional[str] = None) -> List[dict]:
        """
        Detects logos in an image and classifies them.

        Args:
            image_path (str): The path to the input image.
            target_brand (Optional[str]): The target brand to check against, if any.

        Returns:
            List[dict]: A list of dictionaries containing the detected logos and their classifications.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # Detection
        detections = []
        if self.detector:
            with torch.no_grad():
                pred = self.detector(image, conf=self.detector_conf, iou=self.detector_iou, verbose=False)[0]
            if len(pred.boxes) > 0:
                detections = pred.boxes.xyxy.cpu().numpy()
        else:
            # Fallback to XML annotations
            xml_path = os.path.join('data/processed/annotations/val',
                                os.path.splitext(os.path.basename(image_path))[0] + ".xml")
        detections = self._parse_xml_annotations(xml_path)

        results = []
        for det in detections:
            if isinstance(det, np.ndarray):  # For YOLO detections
                x1, y1, x2, y2 = map(int, det[:4])
            else:  # For XML annotations
                x1, y1, x2, y2 = det['bbox']

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # Binary classification
            if target_brand:
                confidence = self.is_target_brand(crop_pil, target_brand)
                brand = target_brand if confidence > 0.5 else "other"
            else:
                brand, confidence = self.classify_logo(crop_pil)

            results.append({
                'brand': brand,
                'confidence': float(confidence),
                'bbox': (x1, y1, x2, y2)
            })

            print(results)

        return results

    def is_target_brand(self, image_crop: Image.Image, target_brand: str, threshold: float = 0.4) -> float:
        """
        Returns the probability that the cropped image belongs to the target brand.

        Args:
            image_crop (Image.Image): The cropped image of the logo.
            target_brand (str): The name of the target brand to check against.
            threshold (float): The threshold for binary classification.

        Returns:
            float: The probability that the cropped image belongs to the target brand [0-1].
        """
        if target_brand not in self.brand_prototypes:
            raise ValueError(f"Target brand {target_brand} not in known prototypes")

        processed_img = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.model.encode_image(processed_img)
            query_emb /= query_emb.norm(dim=-1, keepdim=True)

        # Compute similarity with the target brand
        target_sim = torch.cosine_similarity(query_emb, self.brand_prototypes[target_brand].unsqueeze(0)).item()

        # Normalize to probability using sigmoid
        probability = 1 / (1 + np.exp(-10 * (target_sim - threshold)))

        return probability

    def classify_logo(self, image_crop: Image.Image) -> tuple:
        """
        Classifies the cropped image of the logo.

        Args:
            image_crop (Image.Image): The cropped image of the logo.

        Returns:
            tuple: A tuple containing the brand name and the confidence score.
        """
        processed_img = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.model.encode_image(processed_img)

        query_emb /= query_emb.norm(dim=-1, keepdim=True)
        similarities = {}
        for brand, proto in self.brand_prototypes.items():
            sim = torch.cosine_similarity(query_emb, proto.unsqueeze(0))
            similarities[brand] = sim.item()

        max_brand = max(similarities, key=similarities.get)
        return max_brand, similarities[max_brand]

    def _parse_xml_annotations(self, xml_path: str):
        """
        Parses an XML annotation file.

        Args:
            xml_path (str): The path to the XML annotation file.

        Returns:
            List[dict]: A list of dictionaries containing the annotations.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []
        for obj in root.findall('object'):
            brand = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            annotations.append({
                'brand': brand,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        return annotations

