import pandas as pd
from Features.FeatureID import FeatureID
from Features.IndicatorInterface import IndicatorInterface

class Support(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window: int = 10, min_len_support : int = 50):
        """
        Initialise le détecteur de supports avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close', 'high', 'low', 'volume', et 'date'.
        :param window: Taille de la fenêtre pour détecter les creux locaux.
        :param min_touches: Nombre minimum de touches pour valider un support.
        :param tolerance: Tolérance en pourcentage pour considérer qu'un point touche le support.
        """
        self.data = data
        self.window = window
        self.min_len_support = min_len_support
        self.supports = []

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        """
        Calcule les niveaux de support et les points à tracer.
        """
        if 'low' not in self.data or 'close' not in self.data:
            raise ValueError("Erreur dans Support : Les colonnes 'low' et 'close' doivent être présentes dans les données.")

        # Détection des creux locaux (minima locaux)
        lows = self.data['low'].values
        local_minima = []
        
        for i in range(self.window, len(lows) - self.window):
            if lows[i] == min(lows[i-self.window:i+self.window+1]):
                local_minima.append((i, lows[i]))

        if not local_minima:
            return

        supports = []

        for idxStart, priceStart in local_minima:
            for idxEnd, priceEnd in local_minima:
                if idxStart < (idxEnd - self.min_len_support):

                    a = (priceEnd - priceStart) / (idxEnd - idxStart)
                    b = priceStart - a * idxStart

                    support_points = {}
                    for i in range(idxStart, idxEnd + 1):
                        support_points[i] = a * i + b

                    score = 0
                    for idx, price in local_minima:
                        if idxStart < idx < idxEnd:
                            distance = abs(support_points[idx] - price)
                            score += 1/ (distance * distance) 
                    
                    supports.append({
                        "Score" : score,
                        "idxStart": idxStart,
                        "idxEnd" : idxEnd,
                        "PriceStart": priceStart,
                        "PriceEnd": priceEnd,
                        "points": support_points,
                    })
        
        self.supports = pd.DataFrame(supports).sort_values(by='Score', ascending=False)
        #self.supports = pd.DataFrame(supports).sort_values(by="points", key=lambda x: x.map(len), ascending=False)
        
    def update(self, new_data: pd.DataFrame):
        """
        Met à jour les supports avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if 'low' not in new_data or 'close' not in new_data:
            raise ValueError("Erreur dans Support : Les colonnes 'low' et 'close' doivent être présentes dans les nouvelles données.")
        
        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.supports = []
        self.data = pd.DataFrame()