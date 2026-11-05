from Targets import *
from Features import *
import pandas as pd


class Target_Plus_1(TargetInterface):

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), *args, **kwargs) -> None:
        self.data = data
        self.target = None
        self.training_data_shift = 0

    def calculate(self, resolved_dependencies: dict[TargetID, TargetInterface] = None) -> None:
        shifted = self.data['close'].shift(-1)
        if self.training_data_shift > 0:
            self.target = shifted.iloc[:-self.training_data_shift]
        else:
            self.target = shifted

        

    def update(self, new_data: pd.DataFrame, training_data_shift : int) -> None:
        if 'close' not in new_data:
            raise ValueError("Erreur dans Target_Plus_1 : La colonne 'close' doit être présente dans les nouvelles données.")
        
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