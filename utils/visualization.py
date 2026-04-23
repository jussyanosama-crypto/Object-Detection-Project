import cv2
import numpy as np

def draw_boxes(image, detections):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image (numpy.ndarray): The image to annotate (BGR format for OpenCV).
        detections (list): List of detection dictionaries.
        
    Returns:
        numpy.ndarray: Annotated image.
    """
    # Create a clean, professional color palette (distinct colors based on class_id)
    # Using random colors that are consistent across runs for the same ID
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    # Adaptive thickness and scaling based on image resolution
    height, width = image.shape[:2]
    thickness = max(1, int(min(height, width) / 400))
    font_scale = max(0.4, min(height, width) / 1000)
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        cls_id = det['class_id']
        cls_name = det['class_name']
        
        # Get color for this class
        color = colors[cls_id % len(colors)].tolist()
        
        # 1. Draw Bounding Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 2. Draw Label Text
        label = f"{cls_name} {conf:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Background rectangle for text (filled)
        # Prevent text from going outside image bounds
        text_y = max(y1, text_height + 5)
        
        cv2.rectangle(
            image, 
            (x1, text_y - text_height - 5), 
            (x1 + text_width + 5, text_y + baseline - 5), 
            color, 
            -1 # filled
        )
        
        # Text (white or black depending on background brightness for readability)
        cv2.putText(
            image, 
            label, 
            (x1 + 2, text_y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )
        
    return image
