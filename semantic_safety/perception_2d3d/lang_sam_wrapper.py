import torch
import cv2
from PIL import Image
from transformers import pipeline
from transformers.utils import logging

# Suppress annoying Hugging Face warnings
logging.set_verbosity_error()

class SemanticPerception:
    def __init__(self):
        print("Loading OWL-ViT (Zero-Shot Object Detection)...")
        # Automatically use GPU if available
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load the stable, native Hugging Face pipeline
        self.detector = pipeline(
            model="google/owlvit-base-patch32",
            task="zero-shot-object-detection",
            device=self.device
        )
        print("Model loaded successfully.")

    def detect_objects(self, image_path: str, candidate_labels: list):
        # The pipeline expects a PIL Image
        image_pil = Image.open(image_path).convert("RGB")
        
        # Run the detection
        results = self.detector(
            image_pil,
            candidate_labels=candidate_labels,
        )
        
        # Lower this to 0.05 (5%) so we can see what the model is thinking
        confident_results = [res for res in results if res['score'] > 0.01]
        return confident_results

def run_owlvit_perception():
    image_path = "test_rgb.png"
    target_objects = ["bowl", "red bowl", "dish", "red object", "power drill"]
    
    detector = SemanticPerception()
    
    print(f"\nScanning '{image_path}' for: {target_objects}...")
    results = detector.detect_objects(image_path, target_objects)
    
    if len(results) == 0:
        print("AI failed to find any objects matching the prompts.")
        return
        
    print(f"Success! Found {len(results)} objects.")
    
    # Visualization using standard OpenCV
    img_cv = cv2.imread(image_path)
    
    for result in results:
        label = result['label']
        score = result['score']
        box = result['box']
        
        print(f" - Found '{label}' with {score*100:.1f}% confidence")
        
        x_min, y_min = box['xmin'], box['ymin']
        x_max, y_max = box['xmax'], box['ymax']
        
        # Draw the bounding box (Green for drill, Cyan for bowl)
        color = (0, 255, 0) if label == "power drill" else (255, 255, 0)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add the text label
        text = f"{label}: {score:.2f}"
        cv2.putText(img_cv, text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.imwrite("test_perception.png", img_cv)
    print("\nSaved visualization to 'test_perception.png'. Open it in Cursor!")

if __name__ == "__main__":
    run_owlvit_perception()