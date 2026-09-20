# DR — TRAINING RESULTS

## Experiment

Model        : ConvNeXt-Tiny (convnext_tiny.fb_in22k_ft_in1k)
Task         : 5-class classification
Image size   : 448 × 448
Batch size   : 16
Epochs       : 10
Cross-val    : 5-fold
Device       : CUDA
GPU          : NVIDIA Tesla T4 (15.64 GB)

## Dataset

Total images used : 11,243

Sources:
EyePACS : 7,581
APTOS   : 3,662

Class distribution:
Class 0 : 3,805
Class 1 : 2,370
Class 2 : 2,999
Class 3 : 1,066
Class 4 : 1,003

No missing EyePACS images were detected.
No dataset paths were removed during loading.

## Model size

Backbone parameters : 27,820,128
Classification head : 6,921

## 5-FOLD CROSS-VALIDATION

Fold    Best QWK    Accuracy    Macro F1

---

1       0.7908      0.6065      0.6096
2       0.8016      0.6327      0.6216
3       0.8066      0.6234      0.6211
4       0.7884      0.5992      0.5935
5       0.7931      0.6206      0.6156

Mean    0.7961      0.6165      0.6123

## TRAINING BEHAVIOR

The model showed a consistent improvement in validation QWK during
training across all five folds.

Representative progression:

Fold 1:
Epoch 1 : QWK 0.5853
Epoch 5 : QWK 0.7833
Epoch 6 : QWK 0.7908
Epoch 10: QWK 0.7890

Fold 2:
Epoch 1 : QWK 0.5553
Epoch 5 : QWK 0.7704
Epoch 8 : QWK 0.7980
Epoch 9 : QWK 0.8016
Epoch 10: QWK 0.8007

Fold 3:
Epoch 1 : QWK 0.3411
Epoch 5 : QWK 0.7894
Epoch 8 : QWK 0.8059
Epoch 10: QWK 0.8066

Fold 4:
Epoch 1 : QWK 0.5851
Epoch 5 : QWK 0.7852
Epoch 8 : QWK 0.7884
Epoch 10: QWK 0.7842

Fold 5:
Epoch 1 : QWK 0.5371
Epoch 5 : QWK 0.7835
Epoch 8 : QWK 0.7884
Epoch 10: QWK 0.7931

## FINAL FOLD CLASSIFICATION DETAIL

The final fold (Fold 5) produced:

```
          Precision   Recall   F1
```

Class 0         0.8316    0.6491   0.7292
Class 1         0.4548    0.6477   0.5344
Class 2         0.6133    0.5693   0.5905
Class 3         0.5000    0.5701   0.5328
Class 4         0.7318    0.6550   0.6913

Accuracy                           0.6206
Macro F1                          0.6156
Weighted F1                       0.6291

## FOLD 5 CONFUSION MATRIX

```
          Predicted
         0    1    2    3    4
```

Actual 0    494  222   38    2    5
Actual 1     71  307   88    2    6
Actual 2     27  128  341   84   19
Actual 3      0   11   63  122   18
Actual 4      2    7   26   34  131

## TRAINING / CHECKPOINTING

The training process used:

* Separate learning rates for classification head and backbone
* Class-weighted training
* Best-QWK checkpoint selection
* Early-stopping monitoring
* 5-fold cross-validation

Initial learning rates:
Head     : 1.0e-4
Backbone : 2.0e-5

Learning rates were progressively reduced during training.

## COMPUTE

Hardware : NVIDIA Tesla T4
GPU RAM  : 15.64 GB
Image    : 448 × 448
Batch    : 16

Approximate total wall-clock time from the recorded run:
~26,088 seconds
~7.25 hours

## RESULT SUMMARY

5-fold cross-validation produced:

Mean QWK      : 0.7961
Mean Accuracy : 0.6165
Mean Macro F1 : 0.6123

Best individual fold:
Fold 3
QWK : 0.8066


