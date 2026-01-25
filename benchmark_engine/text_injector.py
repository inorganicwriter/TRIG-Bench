from PIL import Image, ImageDraw, ImageFont
import random
import os
import torch

class VisualAttacker:
    def __init__(self, font_path="arial.ttf", use_yolo=True):
        self.font_path = font_path
        # Try to locate a default font
        if not os.path.exists(self.font_path):
            if os.path.exists("C:\\Windows\\Fonts\\arial.ttf"):
                self.font_path = "C:\\Windows\\Fonts\\arial.ttf"
            elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                 self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        
        self.yolo_model = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                print("Loading YOLOv8n for Object Dissociation Level...")
                self.yolo_model = YOLO("yolov8n.pt") 
            except Exception as e:
                print(f"Warning: Could not load YOLO: {e}. Object targeting will be disabled.")

    def inject_text(self, image: Image.Image, text: str, position="random", target_class=None):
        """
        Injects text onto the image.
        :param position: 'random', 'center', or 'object' (needs target_class)
        :param target_class: list of class names to target (e.g. ['car', 'bus'])
        """
        img = image.convert("RGBA")
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        # Dynamic font size base
        base_size = max(20, int(img.height * 0.1))
        
        # Determine Position
        w, h = img.size
        x, y = 50, 50
        
        box_found = False
        
        # Strategy: Object Targeting
        if position == "object" and self.yolo_model and target_class:
            try:
                # Run inference
                results = self.yolo_model(image, verbose=False) 
                
                # Filter boxes by class
                candidates = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.yolo_model.names[cls_id]
                        if cls_name in target_class:
                            # Save box (x1, y1, x2, y2) and area
                            coords = box.xyxy[0].cpu().numpy()
                            area = (coords[2]-coords[0]) * (coords[3]-coords[1])
                            candidates.append((coords, area))
                
                if candidates:
                    # Pick largest object
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best_box = candidates[0][0] # [x1, y1, x2, y2]
                    
                    # Calculate center of box
                    obj_w = best_box[2] - best_box[0]
                    obj_h = best_box[3] - best_box[1]
                    
                    # Adjust font size to fit object? 
                    # Let's keep it readable but maybe scaled to object width
                    base_size = max(15, int(obj_w * 0.3)) 
                    
                    x = best_box[0] + (obj_w * 0.1) # Slight offset
                    y = best_box[1] + (obj_h * 0.2)
                    box_found = True
                    print(f"      [VisualAttacker] Targeted object: {target_class} at {int(x)},{int(y)}")
                else:
                    print(f"      [VisualAttacker] No {target_class} found. Falling back to random.")
            except Exception as e:
                print(f"      [VisualAttacker] Object detection failed: {e}")

        # Fallback to random if object not found or requested
        if not box_found:
            if position == "center":
                # Temp font load to measure
                font = self._load_font(base_size)
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = right - left, bottom - top
                x = (w - text_w) // 2
                y = (h - text_h) // 2
            else: # Random
                font = self._load_font(base_size)
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = right - left, bottom - top
                max_x = max(0, w - text_w - 10)
                max_y = max(0, h - text_h - 10)
                x = random.randint(10, max_x) if max_x > 10 else 10
                y = random.randint(10, max_y) if max_y > 10 else 10

        # Draw
        final_font = self._load_font(base_size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=final_font)
        text_w, text_h = right - left, bottom - top
        
        # Background Box
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=(0, 0, 0, 160) 
        )
        # Text
        draw.text((x, y), text, font=final_font, fill=(255, 255, 255, 255))

        out = Image.alpha_composite(img, txt_layer)
        return out.convert("RGB")

    def _load_font(self, size):
        try:
            return ImageFont.truetype(self.font_path, int(size))
        except:
            return ImageFont.load_default()

