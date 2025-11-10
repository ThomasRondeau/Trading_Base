import pandas as pd
from typing import Dict, Set, Type, Any
from Features.TargetID import TargetID
from Targets.TargetInterface import TargetInterface


class TargetManager:
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):
        self.data = data
        self.data_shift = 0
        self._cache: Dict[TargetID, TargetInterface] = {}
        self._targets: Dict[TargetID, TargetInterface] = {}

    def set_Shift(self, shift: int):
        self.data_shift = shift

    def register_target(self, target_id : TargetID, target: TargetInterface):
        """
        Enregistre directement une instance de target.
        """
        if target_id in self._targets:
            raise ValueError(f"Target {target_id} already registered.")
        
        self._targets[target_id] = target

    def get_target(self, target_id : TargetID) -> TargetInterface:
        """
        Retourne la valeur de la target demandée.
        """
        if target_id in self._cache:
            return self._cache[target_id]
        
        if target_id not in self._targets:
            raise ValueError(f"Target {target_id} not registered.")
        
        target = self._targets[target_id]

        resolved_dependencies = {}
        for dep_id in target.dependencies:
            resolved_dependencies[dep_id] = self.get_target(dep_id)
            
        target.calculate(resolved_dependencies)
        self._cache[target_id] = target
        return target

    def get_targets(self, target_ids : Set[TargetID]) -> Dict[TargetID, TargetInterface]:
        """
        Retourne un dictionnaire de valeurs pour les targets demandées.
        """
        return {tid: self.get_target(tid) for tid in target_ids if tid in self._targets}
    

    def update(self, new_data: pd.DataFrame, training_data_shift: int = 0) -> None:
        
        self.data = new_data
        self.clear_cache()
        self.data_shift = training_data_shift
        print(f"Updating TargetManager with training_data_shift = {training_data_shift}") #TODO remove
        for target in self._targets.values():
            target.update(new_data, training_data_shift) 


    def clear_cache(self) -> None:
        self._cache.clear()
        for target in self._targets.values():
            target.clear_cache()
