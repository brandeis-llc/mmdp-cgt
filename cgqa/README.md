## CGT Evaluation
### Accumulative F1
Given a dialogue with N accepted statements on the block weight, we
evaluate the quality of the generated answers using accumulative precision, recall and F1.

For each accepted statement, we evaluate the predicted common ground on the block weight 
by comparing it with the ground truth common ground that has been established from the beginning of the dialogue to the timestamp of current statement.

We show the evaluation results with the average of PRF1 scores over all accepted statements, and the 
PRF1 scores for the final accepted statement in the dialogue.

### Dice-Sørensen coefficient (DSC)
The same metric used in the common ground tracking paper.
DSC indicates how much the set of propositions extracted by the model matches the set of propositions in the ground truth.

It is more stringent than the accumulative F1, as it cannot share the predicted proposition from previous dialogue.

### CGT Evaluation Example
```
row 1: ground truth common ground
row 2: predicted common ground
row 3: precision, recall, f1
row 4: DSC

{'red': '10', 'blue': '10'}
{'red': '10', 'blue': '10'}
1.0 1.0 1.0
1.0
---------------------------------
{'red': '10', 'blue': '10'}
{}
1.0 1.0 1.0  # the weight of red and blue is already accepted from last statement, so the score is still 1
0.6666666666666666
---------------------------------
{'red': '10', 'blue': '10', 'green': '20'}
{'red': '10', 'blue': '10', 'green': '20'}
1.0 1.0 1.0
0.8333333333333334
---------------------------------
{'blue': '10', 'red': '10', 'purple': '30', 'green': '20'}
{'red': '10', 'blue': '30', 'green': '20', 'purple': '30'}
0.75 1.0 0.8571428571428571  # blue is updated from 10 to 30, which results in a false positive
0.8
---------------------------------
{'yellow': '50', 'green': '20', 'blue': '10', 'purple': '30', 'red': '10'}
{'red': '10', 'blue': '30', 'green': '20', 'purple': '30'}
0.75 0.75 0.75  # yellow is missing, which results in a false negative
0.7586206896551724
---------------------------------
average PRF1: 0.9166666666666666 0.9583333333333334 0.9345238095238095
final PRF1: 0.75 0.75 0.75
average DSC: 0.8114
final DSC: 0.7586206896551724
```