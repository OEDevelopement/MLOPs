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
    param_grid_lr = {"l1_ratio": 0.5}  # Wird für `elasticnet` benötigt

    # ✅ Teste alle gültigen Kombinationen
    valid_lr_params = {
        "set_1": ("l2", 1.0, "lbfgs", 100),
        "set_2": ("l1", 0.5, "liblinear", 200),
        "set_3": ("elasticnet", 0.7, "saga", 150),
        "set_4": (None, 1.0, None, 100)
    }

    for key, params in valid_lr_params.items():
        assert is_valid_lr_params(params, param_grid_lr) == True, f"Fehlgeschlagen für {key}: {params}"

    # ❌ Teste alle ungültigen Kombinationen
    invalid_lr_params = {
        "set_1": ("l1", 1.0, "lbfgs", 100),  # lbfgs unterstützt kein l1
        "set_2": ("elasticnet", 0.5, "liblinear", 200),  # liblinear unterstützt kein elasticnet
        "set_3": ("l1", 0.5, None, 150), # `None` als Solver unterstützt `l1` nicht (nur `None` als Penalty erlaubt)
        "set_4": ("elasticnet", 1.0, "saga", 100)  # l1_ratio fehlt in param_grid_lr
    }

    for key, params in invalid_lr_params.items():
        assert is_valid_lr_params(params, {}) == False, f"Fehlgeschlagen für {key}: {params}"

# Teste die Funktion select_diverse_combinations, um sicherzustellen, dass sie die richtige Anzahl an Parametern auswählt
def test_select_diverse_combinations():
    param_combinations = list(itertools.product(*param_grid_lr.values()))
    selected_combinations = select_diverse_combinations(param_combinations, 3)
    assert len(selected_combinations) == 3  # Maximale Anzahl von 3
    assert len(selected_combinations) == len(set(selected_combinations)) # Es gibt Duplikate in den ausgewählten Kombinationen
