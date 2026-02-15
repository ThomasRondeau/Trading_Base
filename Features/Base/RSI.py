import pandas as pd
from Features import *

class RSI(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), window: int = 14):
        """
        Initialise le calculateur de RSI avec les données fournies.

        :param data: DataFrame contenant les colonnes 'close' et 'date'.
        :param period: Période pour le calcul du RSI.
        """
        self.data = data
        self.window = window
        self.rsi = None

    def calculate(self, resolved_dependencies: dict[FeatureID, FeatureInterface] = None):
        """
        Calcule le RSI et l'ajoute à l'attribut `self.rsi`.
        """
        if 'close' not in self.data:
            raise ValueError("Erreur dans RSI : La colonne 'close' doit être présente dans les données.")

        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()

        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))

    def update(self, new_data: pd.DataFrame):
        """
        Met à jour le RSI avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if 'close' not in new_data:
            raise ValueError("Erreur dans RSI : La colonne 'close' doit être présente dans les nouvelles données.")
        
        self.data = new_data
    
    def clear_cache(self):
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.rsi = None
        self.data = pd.DataFrame()