import string
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd


@dataclass(frozen=True)
class GridSearchCVTuningResult:
    """
    test_case: What test case the result is associated with
    best_score: The best ROC AUC (Receiver Operating Characteristic Area Under the Curve) from the tuning run
    best_params: the best Hyperparameter combination
    cv_results: cv data as a whole
    """
    test_case: string
    best_score: float
    best_params: Dict[str, Any]
    cv_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case": self.test_case,
            "best_score": self.best_score,
            "best_params": self.best_params
        }

    def __str__(self) -> str:
        # Pretty-formatted string for printing
        param_str = "\n".join([f"  {k}: {v}" for k, v in self.best_params.items()])
        return (
            f"[Tuning Result]:[{self.test_case}]\n"
            f"Best Score: {self.best_score:.4f}\n"
            f"Best Hyperparameters:\n{param_str}"
        )

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        if self.cv_results:
            return pd.DataFrame(self.cv_results)
        return None

    def export_csv(self, path: str = "tuning_results.csv"):
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(path, index=False)
