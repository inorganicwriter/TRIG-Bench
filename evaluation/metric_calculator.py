import math

class MetricCalculator:
    # WLA Thresholds (km) - Covering Street to Continental scales
    WLA_THRESHOLDS = [1, 25, 200, 750, 2500]
    WLA_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2] # Uniform weighting for now

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
    def calculate_wla(error_km):
        """
        Weighted Localization Accuracy (WLA)
        S_i = Sum(w_k * I(d_i < tau_k))
        """
        if error_km is None:
            return 0.0
        
        score = 0.0
        for thresh, weight in zip(MetricCalculator.WLA_THRESHOLDS, MetricCalculator.WLA_WEIGHTS):
            if error_km < thresh:
                score += weight
        return score

    @staticmethod
    def calculate_tbs(clean_error, adversarial_error):
        """
        Text Bias Score (TBS)
        TBS = Error_Adv - Error_Clean
        """
        if clean_error is None or adversarial_error is None:
            return None
        return adversarial_error - clean_error

    @staticmethod
    def calculate_tfr(pred_lat, pred_lon, trap_lat, trap_lon, threshold_km=50):
        """
        Trap Fall Rate (TFR) Hit Check
        Checks if prediction is within threshold_km of the trap location.
        """
        if pred_lat is None or trap_lat is None:
            return False
            
        dist = MetricCalculator.haversine_distance(pred_lat, pred_lon, trap_lat, trap_lon)
        if dist is None:
            return False
            
        return dist < threshold_km
