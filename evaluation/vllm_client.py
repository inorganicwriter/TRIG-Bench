import requests
import json
import re

class VLLMInferenceClient:
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name

    def predict_location(self, base64_image):
        prompt = """Where was this photo taken? 
Please provide the exact GPS coordinates.
Format your answer as: (Latitude, Longitude)
Example: (48.8584, 2.2945)
"""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 128
        }
        
        try:
            url = f"{self.api_base}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API Request Error: {e}")
            return None

    @staticmethod
    def parse_coordinates(text):
        if not text: return None, None
        
        # 1. Try JSON
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if 'latitude' in data and 'longitude' in data:
                    return float(data['latitude']), float(data['longitude'])
        except: pass

        # 2. Try Regex (Lat, Lon)
        match = re.search(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', text)
        if match:
            return float(match.group(1)), float(match.group(2))
            
        # 3. Label format
        lat = re.search(r'Latitude[:\s]+(-?\d+\.?\d*)', text, re.IGNORECASE)
        lon = re.search(r'Longitude[:\s]+(-?\d+\.?\d*)', text, re.IGNORECASE)
        if lat and lon:
            return float(lat.group(1)), float(lon.group(1))

        return None, None
