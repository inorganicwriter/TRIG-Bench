import math

class MetricCalculator:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance (km) between two points.
        """
        try:
            # Convert decimal degrees to radians 
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a)) 
            r = 6371 # Radius of earth in kilometers
            return c * r
        except Exception:
            return None

    @staticmethod
    def calculate_text_bias_score(clean_error, adversarial_error):
        """
        Calculates how much the text distracted the model.
        Score > 0 means adversarial text increased error.
        """
        if clean_error is None or adversarial_error is None:
            return None
        return adversarial_error - clean_error
