import os
import cv2
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    print("🚀 Initializing YOLO Inference...")
    
    # Load model
    model_path = os.path.join("model", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
        
    detector = YOLOModel(model_path)
    
    # Load demo image
    image_path = os.path.join("assets", "demo.png")
    if not os.path.exists(image_path):
        print(f"❌ Error: Demo image not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return
        
    # Run inference
    print("🔍 Running inference...")
    detections = detector.predict(image)
    
    # Print detections
    print("\n📊 Detections:")
    for d in detections:
        print(f"- {d['class_name']} ({d['confidence']:.2f}) at {d['box']}")
        
    # Visualize and save output
    print("\n🎨 Generating annotated image...")
    annotated_image = draw_boxes(image.copy(), detections)
    
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    main()
