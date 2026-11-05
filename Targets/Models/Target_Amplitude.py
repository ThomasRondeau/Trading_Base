from Targets import *
from Features import *
import pandas as pd


class Target_Amplitude(TargetInterface):

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), *args, **kwargs) -> None:
        self.data = data
        self.target = None
        self.training_data_shift = 0

    def calculate(self, resolved_dependencies: dict[TargetID, TargetInterface] = None) -> None:
        # Target for index i is (price[i+5] - price[i]) / price[i]
        close = self.data['close']
        shifted = close.shift(-5)
        
        full_target = ((shifted - close) / close) * 100
        
        # On ne garde que la partie qui correspond aux features (avant le shift de sécurité)
        if self.training_data_shift > 0:
            self.target = full_target.iloc[:-self.training_data_shift]
        else:
            self.target = full_target
        

    def update(self, new_data: pd.DataFrame, training_data_shift : int) -> None:
        if 'close' not in new_data:
            raise ValueError("Erreur dans Target_Amplitude : La colonne 'close' doit être présente dans les nouvelles données.")
        
        self.data = new_data
        self.training_data_shift = training_data_shift

    def clear_cache(self) -> None:
        """
        Vide le cache de l'indicateur.
        Utilisé pour réinitialiser l'état de l'indicateur.
        """
        self.target = None
        self.data = pd.DataFrame()
        self.training_data_shift = 0