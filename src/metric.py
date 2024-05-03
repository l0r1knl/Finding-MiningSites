import torch
import torchmetrics
from torchmetrics.functional import f1_score
from torch import Tensor


class OptimalF1Score(torchmetrics.Metric):
    """Compute `F1 score` searching for threshold at which F1 score is maximized.
    """

    def __init__(self):
        
        super().__init__()
        self.add_state(
            "preds",
            default=torch.tensor([]),
            dist_reduce_fx="cat"
        )
        self.add_state(
            "target",
            default=torch.tensor([]),
            dist_reduce_fx="cat"
        )
        self.task = "binary"
        self.threshold = 0.0

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        
        self.threshold = 0.0
        return super().reset()

    def compute(self) -> Tensor:
        """Compute metric and search optimal threshold.

        Returns:
            Tensor: F1 Score computed with optimal thresholds.
        """

        best_score, best_threshold = 0.0, -1.0
        for i in range(100):
            threshold = i / 100
            _score = f1_score(self.preds > threshold, self.target, self.task)

            if _score > best_score:
                best_score = _score
                best_threshold = threshold

        self.threshold = best_threshold

        return torch.tensor(best_score)
