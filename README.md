# Summary Table

| Training Instance  | Optimizer Used  | Reguralizer Used  | Epochs  | Early Stopping  | Number Of Layers  | Learning Rate  |Accuracy  | Recall  | F1 Score | Precision|
|--------            |--------         |--------           |-------- |--------         |--------           |--------        |--------  |-------- |--------  |--------  |
| Instance 1         | None            | None              | 10      | no              | 5                 | Default        | 95%      | 0.18    | 0.25     | 0.25     |
| Instance 2         | adam            | l2                | 20      | yes             | 5                 | 0.0005         | 96%      | 0.25    | 0.35     | 0.57     |
| Instance 3         | RMSPROP         | l2                | 20      | yes             | 5                 | 0.0001         | 95%      | 0.11    | 0.19     | 0.62     |
| Instance 4         | nadam           | l2                | 20      | yes             | 5                 | 0.0002         | 96%      | 0.13    | 0.23     | 0.66     |
| Instance 5         | adagrad         | l1_l2             | 30      | yes             | 5                 | 0.001          | 95%      | 0.06    | 0.12     | 0.75     |
