import os
import torch
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        """
        Initialize the YOLO model, set device automatically, and load class names.
        """
        # Determine device: CUDA if available, else CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚙️ Loading model on: {self.device.upper()}")
        
        # Load YOLO model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Load labels
        self.class_names = {}
        self._load_labels(labels_path)
        
    def _load_labels(self, labels_path):
        """Dynamically load labels from labels.txt."""
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                for idx, label in enumerate(lines):
                    self.class_names[idx] = label
        else:
            print(f"⚠️ Warning: {labels_path} not found. Using model's internal classes.")
            self.class_names = self.model.names
            
    def predict(self, image, conf=0.25):
        """
        Run inference on an image.
        Accepts: PIL image or numpy array (BGR or RGB depending on caller, YOLO handles PIL and np arrays natively).
        Returns: Structured list of detections.
        """
        # Run inference
        results = self.model.predict(image, conf=conf, device=self.device, verbose=False)
        
        detections = []
        
        # Parse results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                # Class ID
                class_id = int(box.cls[0].cpu().numpy())
                
                # Class Name
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return detections
