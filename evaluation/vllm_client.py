import requests
import json
import re

class VLLMInferenceClient:
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name

    def _clean_content(self, text):
        """Clean thinking tags and markdown."""
        if not text: return ""
        # Remove <think> tags
        if "</think>" in text:
            text = text.split("</think>")[-1]
        elif "<think>" in text:
             text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        
        # Remove markdown code blocks
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        return text.strip()

    def predict_location(self, base64_image):
        prompt = """Analyze this photo and determine where it was taken.
You MUST provide your best estimate of GPS coordinates even if uncertain.
Do NOT refuse. Always give a coordinate guess.
Output ONLY: (Latitude, Longitude)
Example: (48.8584, 2.2945)
"""
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        max_retries = 3
        current_temp = 0.0
        
        for attempt in range(max_retries + 1):
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
                "temperature": current_temp,
                "max_tokens": 1024,
                "frequency_penalty": 0.1
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "choices" not in result or not result["choices"]:
                    print(f"API Error: No choices returned.")
                    continue
                    
                choice = result["choices"][0]
                finish_reason = choice.get("finish_reason")
                
                # 1. Runaway Check (Thinking model truncation)
                if finish_reason == "length":
                    if attempt < max_retries:
                        print(f"  ⚠️ Thinking runaway (truncated), retrying {attempt+1}/{max_retries}...")
                        current_temp = min(current_temp + 0.1, 1.0)
                        continue
                    else:
                        print("  ❌ Max retries for runaway.")
                        return None
                
                content = choice["message"].get("content", "")
                
                # 2. Clean Content (remove <think> tags)
                cleaned = self._clean_content(content)
                
                # 3. Validation: Try parsing coordinates
                if cleaned:
                    test_lat, test_lon = self.parse_coordinates(cleaned)
                    
                    if test_lat is not None:
                        # Successfully parsed - return
                        return cleaned
                    else:
                        # Coordinates not found - retry with higher temperature
                        if attempt < max_retries:
                            print(f"  ⚠️ No coordinates found, retrying {attempt+1}/{max_retries}...")
                            current_temp = min(current_temp + 0.15, 1.0)
                            continue
                        else:
                            # Last attempt - return whatever we have
                            return cleaned
                
                # Empty content
                if attempt < max_retries:
                    print(f"  ⚠️ Empty response, retrying...")
                    continue

            except Exception as e:
                print(f"API Request Error (Attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    import time
                    time.sleep(1)
                    continue
                    
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
