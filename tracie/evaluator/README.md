## TRACIE Evaluator

This script evaluates NLI predictions against correct inferences and produces 4 accuracy scores described below, and can be used to check that outputs produced for the leaderboard are well formed. 

## Example
```
%python3 evaluator/evaluator.py --question_answers data_public/train_iid.jsonl --predictions data_public/train_iid_predictions.txt  --output metrics.json --train_type train_iid

% cat metrics.json
{"train_type": "train_iid", "total_acc": 0.5, "start_acc": 0.5, "end_acc": 0.5, "story_em": 0.0}
```

This uses a dummy train prediction file called `train_iid_predictions.txt` from the training file `train_iid.jsonl` (which predicts entailments for each example), which consists of one label prediction for line:
```
entailment
entailment
entailment
entailment
```

## Output metrics

A json file called `metrics.json` will be produced containing the following accuracy scores:
```
{"train_type": "train_iid", "total_acc": 0.5, "start_acc": 0.5, "end_acc": 0.5, "story_em": 0.0}
```
`total_acc` is the overall accuracy; `start_acc` is the accuracy of  the subset of problems involving event `start` questions; `end_acc` is the subset involving end point questions and; `story_em` is the accuracy of getting all questions correct per story. `train_type` is the particular training set used to obtain the result.
