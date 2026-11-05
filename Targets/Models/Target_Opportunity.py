from Targets import *
from Features import *
import pandas as pd


class Target_Opportunity(TargetInterface):

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), threshold_pct: float = 0.05, horizon: int = 5) -> None:
        """
        Opportunité = |variation du close à `horizon` barres| > `threshold_pct` (en %, 0.05 = 0,05 %).
        """
        self.data = data
        self.target = None
        self.training_data_shift = 0
        self.threshold_pct = threshold_pct
        self.horizon = horizon

    def calculate(self, resolved_dependencies: dict[TargetID, TargetInterface] = None) -> None:
        close = self.data['close']
        shifted = close.shift(-self.horizon)

        amplitude_pct = ((shifted - close) / close) * 100
        full_target = (amplitude_pct.abs() > self.threshold_pct).astype(int)

        if self.training_data_shift > 0:
            self.target = full_target.iloc[:-self.training_data_shift]
        else:
            self.target = full_target

        

    def update(self, new_data: pd.DataFrame, training_data_shift : int) -> None:
        if 'close' not in new_data:
            raise ValueError("Erreur dans Target_Opportunity : La colonne 'close' doit être présente dans les nouvelles données.")
        
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