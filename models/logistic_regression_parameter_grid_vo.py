from dataclasses import dataclass
from typing import List, Dict


# Define an immutable value object (VO) for the Logistic Regression hyperparameter grid
@dataclass(frozen=True)
class LRParameterGrid:
    # List of regularization strength values to try.
    # Lower values mean stronger regularization (e.g., C=0.01 penalizes large coefficients more than C=10).
    C_values: List[float]

    # List of penalty types to use for regularization.
    # 'l1' encourages sparsity (feature selection), 'l2' is standard ridge-style regularization.
    penalties: List[str]

    # List of solver algorithms to use for optimization.
    # E.g., 'liblinear' works for small datasets and supports both 'l1' and 'l2' penalties.
    solvers: List[str]

    # Converts the value object into a dictionary that is compatible with sklearn's GridSearchCV
    def to_grid(self) -> Dict[str, List]:
        return {
            'C': self.C_values,
            'penalty': self.penalties,
            'solver': self.solvers
        }
