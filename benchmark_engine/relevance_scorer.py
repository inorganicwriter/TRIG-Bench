import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class SemanticRelevanceEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model ({model_name}) on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise e

    def calculate_similarity(self, image: Image.Image, candidate_texts: list[str]):
        """
        Calculates cosine similarity between an image and a list of candidate texts.
        """
        # Preprocess
        # CLIP expects "A photo of {text}" usually works better, but raw text is fine too.
        # We assume caller handles prompt engineering if needed.
        inputs = self.processor(
            text=candidate_texts, 
            images=image, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate Cosine Similarity
        # outputs.image_embeds: [1, embed_dim]
        # outputs.text_embeds:  [n_texts, embed_dim]
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        # [1, embed_dim] @ [embed_dim, n_texts] -> [1, n_texts]
        cosine_scores = (image_embeds @ text_embeds.t()).squeeze(0).cpu().numpy()

        return dict(zip(candidate_texts, cosine_scores))

    def classify_distractors(self, image: Image.Image, distractor_candidates: list[str]):
        """
        Classifies distractors into Hard, Mid, Easy levels based on similarity score.
        """
        # Add prompt engineering
        prompts = [f"A photo of {city}" for city in distractor_candidates]
        # Map back to original city name
        prompt_to_city = {p: c for p, c in zip(prompts, distractor_candidates)}
        
        scores = self.calculate_similarity(image, prompts)
        
        categorized = {
            "hard": [],
            "mid": [],
            "easy": []
        }
        
        for text, score in scores.items():
            original_city = prompt_to_city[text]
            # Thresholds can be tuned
            if score > 0.28: 
                categorized["hard"].append((original_city, float(score)))
            elif score > 0.20:
                categorized["mid"].append((original_city, float(score)))
            else:
                categorized["easy"].append((original_city, float(score)))
                
        # Sort by score descending
        for level in categorized:
            categorized[level].sort(key=lambda x: x[1], reverse=True)
            
        return categorized
