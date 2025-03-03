import pytest
import itertools
from param_grid_functions import is_valid_lr_params, select_diverse_combinations

# Beispielwerte für Parametergitter
param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", None],
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs", "saga"],
    "max_iter": [100, 200, 500]
}

# Teste, ob die Funktion is_valid_lr_params korrekt funktioniert
def test_is_valid_lr_params():
    valid_params = ("l1", 0.1, "liblinear", 100)
    assert is_valid_lr_params(valid_params, param_grid_lr) == True
    
    invalid_params = ("l1", 0.1, "unknown_solver", 100)
    assert is_valid_lr_params(invalid_params, param_grid_lr) == False

# Teste die Funktion select_diverse_combinations, um sicherzustellen, dass sie die richtige Anzahl an Parametern auswählt
def test_select_diverse_combinations():
    param_combinations = list(itertools.product(*param_grid_lr.values()))
    selected_combinations = select_diverse_combinations(param_combinations, 3)
    assert len(selected_combinations) == 3  # Maximale Anzahl von 3
    assert all(is_valid_lr_params(params, param_grid_lr) for params in selected_combinations)
