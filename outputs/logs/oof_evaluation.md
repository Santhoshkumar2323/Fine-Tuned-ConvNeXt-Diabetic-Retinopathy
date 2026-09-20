# DR — Out-of-Fold Evaluation Results

## Evaluation

Total images evaluated: 11,243

The final evaluation combines predictions from all 5 out-of-fold (OOF) splits.

| Fold | Best QWK |
|---|---:|
| Fold 0 | 0.7908 |
| Fold 1 | 0.8016 |
| Fold 2 | 0.8066 |
| Fold 3 | 0.7884 |
| Fold 4 | 0.7931 |

## Final OOF Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.6166 |
| Macro-F1 | 0.6142 |
| QWK | 0.7937 |

## Confusion Matrix

Rows represent the true class and columns represent the predicted class.

```text
[[2359 1219  196    3   28]
 [ 315 1590  425   11   29]
 [ 118  655 1681  413  132]
 [   9   55  253  630  119]
 [   3   26  140  162  672]]