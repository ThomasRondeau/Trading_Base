import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Features.FeatureID import FeatureID
from Indicators.IndicatorInterface import IndicatorInterface

class Resistance(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window: int = 10, min_len_resistance : int = 50):
        """
        Initialise le détecteur de résistances avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close', 'high', 'low', 'volume', et 'date'.
        :param window: Taille de la fenêtre pour détecter les pics locaux.
        :param min_touches: Nombre minimum de touches pour valider une résistance.
        :param tolerance: Tolérance en pourcentage pour considérer qu'un point touche la résistance.
        """
        self.data = data
        self.window = window
        self.min_len_resistance = min_len_resistance
        self.resistances = []


    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface] = None):
        if 'high' not in self.data or 'close' not in self.data:
            raise ValueError("Erreur dans Resistance : Les colonnes 'high' et 'close' doivent être présentes dans les données.")

        # Détection des pics locaux (maxima locaux)
        highs = self.data['high'].values
        local_maxima = []
        
        for i in range(self.window, len(highs) - self.window):
            if highs[i] == max(highs[i-self.window:i+self.window+1]):
                local_maxima.append((i, highs[i]))

        if not local_maxima:
            return
        

        resistances = []

        for idxStart, priceStart in local_maxima:
            for idxEnd, priceEnd in local_maxima:
                if idxStart < (idxEnd - self.min_len_resistance):

                    a = (priceEnd - priceStart) / (idxEnd - idxStart)
                    b = priceStart - a * idxStart

                    resistance_points = {}
                    for i in range(idxStart, idxEnd + 1):
                        resistance_points[i] = a * i + b

                    score = 0
                    for idx, price in local_maxima:
                        if idxStart < idx < idxEnd:
                            distance = abs(resistance_points[idx] - price)
                            score += 1/ (distance * distance) 
                    
                    resistances.append({
                        "Score" : score,
                        "idxStart": idxStart,
                        "idxEnd" : idxEnd,
                        "PriceStart": priceStart,
                        "PriceEnd": priceEnd,
                        "points": resistance_points,
                    })
        
        self.resistances = pd.DataFrame(resistances).sort_values(by='Score', ascending=False)
        #self.resistances = pd.DataFrame(resistances).sort_values(by="points", key=lambda x: x.map(len), ascending=False)

    def update(self, new_data: pd.DataFrame):
        if 'high' not in new_data or 'close' not in new_data:
            raise ValueError("Erreur dans Resistance : Les colonnes 'high' et 'close' doivent être présentes dans les nouvelles données.")
        
        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.resistances = []
        self.data = pd.DataFrame()