import os
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from typing import List, Tuple, Dict


class LogoEvaluator:
    """
    A class for evaluating the performance of a few-shot logo recognition model.
    It calculates precision, recall, F1-score, and other metrics for each brand
    and aggregates the results across all brands.

    Args:
        recognizer: An instance of FewShotLogoRecognizer for logo detection and classification.
        support_set (List[Tuple[str, str]]): A list of tuples (image path, brand) for the support set.
        query_set (List[Tuple[str, str]]): A list of tuples (image path, brand) for the query set.
        iou_threshold (float, optional): IoU threshold for bounding box matching. Defaults to 0.4.
        max_size (int, optional): Maximum number of brands to evaluate. Defaults to None (evaluate all).
    """

    def __init__(
        self,
        recognizer,
        support_set: List[Tuple[str, str]],
        query_set: List[Tuple[str, str]],
        iou_threshold: float = 0.4,
        max_size = None
    ):
        """
        Initializes the LogoEvaluator class.

        Args:
            recognizer: An instance of FewShotLogoRecognizer for logo detection and classification.
            support_set (List[Tuple[str, str]]): A list of tuples (image path, brand) for the support set.
            query_set (List[Tuple[str, str]]): A list of tuples (image path, brand) for the query set.
            iou_threshold (float, optional): IoU threshold for bounding box matching. Defaults to 0.4.
            max_size (int, optional): Maximum number of brands to evaluate. Defaults to None (evaluate all).
        """
        self.recognizer = recognizer
        self.support_set = support_set
        self.query_set = query_set
        self.iou_threshold = iou_threshold
        self.brand_metrics = defaultdict(dict)
        self.max_size = max_size

    def evaluate_all_brands(self):
        """
        Evaluates the performance for all brands in the query set.

        Returns:
            dict: Aggregated metrics for all brands, including macro and weighted averages.
        """
        # Group query_set and support_set by brand
        query_by_brand = defaultdict(list)
        for img_path, brand in self.query_set:
            query_by_brand[brand].append(img_path)

        support_by_brand = defaultdict(list)
        for img_path, brand in self.support_set:
            support_by_brand[brand].append(img_path)

        # Calculate metrics for each brand
        if not self.max_size:
            self.max_size = len(query_by_brand)
        for brand, query_images in list(query_by_brand.items())[:self.max_size]:
            print(f"\nEvaluating brand: {brand}")
            support_images = support_by_brand.get(brand, [])
            metrics = self._calculate_brand_metrics(brand, query_images, support_images)
            self.brand_metrics[brand] = metrics

        return self._aggregate_metrics()

    def _calculate_brand_metrics(
        self, brand: str, query_images: List[str], support_images: List[str]
    ):
        """
        Calculates precision, recall, F1-score, and support for a specific brand.

        Args:
            brand (str): The brand to evaluate.
            query_images (List[str]): List of query image paths for the brand.
            support_images (List[str]): List of support image paths for the brand.

        Returns:
            dict: A dictionary containing precision, recall, F1-score, and support for the brand.
        """
        y_true = []
        y_scores = []

        for query_img in tqdm(query_images, desc=f"Processing {brand}"):
            # Get ground truth for the query image
            gt_data = self._parse_xml_annotations(
                os.path.join(
                    "data/processed/annotations/val",
                    os.path.splitext(os.path.basename(query_img))[0] + ".xml",
                )
            )
            has_target = any(ann["brand"] == brand for ann in gt_data)

            # Get predictions for the query image
            preds = self.recognizer.detect_logos(query_img, brand)

            # Find the maximum confidence score for the target brand in predictions
            max_score = max(
                (pred["confidence"] for pred in preds if pred["brand"] == brand),
                default=0.0,
            )

            y_true.append(1 if has_target else 0)
            y_scores.append(max_score)

            # Handle False Negatives using IoU
            if has_target:
                matched = any(
                    self._calculate_iou(gt["bbox"], pred["bbox"]) > self.iou_threshold
                    for gt in gt_data
                    if gt["brand"] == brand
                    for pred in preds
                )
                if not matched:
                    y_scores[-1] = 0.0  # Treat FN as zero confidence

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, [s >= 0.5 for s in y_scores], average="binary", zero_division=0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(y_true),
        }

    def _aggregate_metrics(self):
        """
        Aggregates metrics across all brands to compute macro and weighted averages.

        Returns:
            dict: A dictionary containing metrics for each brand, macro averages, and weighted averages.
        """
        metrics = {
            "brands": {},
            "macro_avg": defaultdict(float),
            "weighted_avg": defaultdict(float),
        }

        total_support = 0
        for brand, brand_metrics in self.brand_metrics.items():
            metrics["brands"][brand] = brand_metrics
            support = brand_metrics["support"]

            for metric in ["precision", "recall", "f1"]:
                metrics["macro_avg"][metric] += brand_metrics[metric]
                metrics["weighted_avg"][metric] += brand_metrics[metric] * support

            total_support += support

        # Normalize averages
        num_brands = len(self.brand_metrics)
        for metric in ["precision", "recall", "f1"]:
            metrics["macro_avg"][metric] /= num_brands if num_brands > 0 else 1
            metrics["weighted_avg"][metric] /= total_support if total_support > 0 else 1

        return metrics

    def _parse_xml_annotations(self, xml_path: str):
        """
        Parses an XML annotation file to extract brand and bounding box information.

        Args:
            xml_path (str): Path to the XML annotation file.

        Returns:
            List[Dict]: A list of dictionaries containing brand and bounding box information.
        """
        if not os.path.exists(xml_path):
            return []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []
        for obj in root.findall("object"):
            brand = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            annotations.append({"brand": brand, "bbox": [xmin, ymin, xmax, ymax]})

        return annotations

    def _calculate_iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) for two bounding boxes.

        Args:
            box1 (List[int]): Coordinates of the first bounding box [xmin, ymin, xmax, ymax].
            box2 (List[int]): Coordinates of the second bounding box [xmin, ymin, xmax, ymax].

        Returns:
            float: The IoU score between the two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (area1 + area2 - inter_area + 1e-6)