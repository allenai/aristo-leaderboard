## TRACIE Evaluator

This script evaluates NLI predictions against correct inferences and produces 4 accuracy scores described below, and can be used to check that outputs produced for the leaderboard are well formed. 

## Example
```
%python3 evaluator/evaluator.py --question_answers data_public/train.jsonl --predictions data_public/train_predictions.txt --output metrics.json

% cat metrics.json
{"total_acc": 0.5012809564474808, "start_acc": 0.500945179584121, "end_acc": 0.5015576323987538, "story_em": 0.0}
```

This uses a dummy train prediction file called `train_predictions.txt` from the training file `train.jsonl` (which predicts entailments for each example), which consists of one label prediction for line:
```
entailment
entailment
entailment
entailment
```

## Output metrics

A json file called `metrics.json` will be produced containing the following accuracy scores:
```
{"total_acc": 0.5012809564474808, "start_acc": 0.500945179584121, "end_acc": 0.5015576323987538, "story_em": 0.0}
```
`total_acc` is the overall accuracy; `start_acc` is the accuracy of the subset of problems involving event `start` questions; `end_acc` is the subset involving end point questions and; `story_em` is the accuracy of getting all questions correct per story.
