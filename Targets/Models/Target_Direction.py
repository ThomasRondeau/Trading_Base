from Targets.TargetInterface import TargetInterface
from Features.TargetID import TargetID
import pandas as pd


class Target_Direction(TargetInterface):

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), *args, **kwargs) -> None:
        self.data = data
        self.target = None
        self.training_data_shift = 0

    def calculate(self, resolved_dependencies: dict[TargetID, TargetInterface] = None) -> None:
        shifted = self.data['close'].iloc[5 : 5 - self.training_data_shift].reset_index(drop=True)
        self.target = (shifted > self.data['close'][:-self.training_data_shift].reset_index(drop=True)).astype(int)
        

    def update(self, new_data: pd.DataFrame, training_data_shift : int) -> None:
        if 'close' not in new_data:
            raise ValueError("Erreur dans Target_Direction : La colonne 'close' doit être présente dans les nouvelles données.")
        
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