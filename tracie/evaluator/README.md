## TRACIE Evaluator

This script evaluates NLI predictions against correct inferences and produces 4
accuracy scores described below, and can be used to check that outputs produced
for the leaderboard are well formed. 

## Example

```bash
% python3 evaluator.py --question_answers ../data/train.jsonl --predictions ../data/train-dummy-predictions.txt --output metrics.json
```

This uses the dummy train prediction file `train-dummy-predictions.txt` which
predicts `entailment` for all examples in `train.jsonl`.

The lines in the prediction file corresponds to the 1171 examples in the labeled
file `train.jsonl`. That is, for example, there are 1171 lines like this:

```
entailment
entailment
entailment
entailment
...
```

## Output metrics

A JSON file called `metrics.json` will be produced containing the accuracy
scores. In this file there are several fields:

* `total_acc` is the overall accuracy
* `start_acc` is the accuracy of the subset of problems involving event `start` questions
* `end_acc` is the subset involving end point questions
* `story_em` is the accuracy of getting all questions correct per story.

For the above example, the `metrics.json` file looks like this:

```js
{"total_acc": 0.5012809564474808, "start_acc": 0.500945179584121, "end_acc": 0.5015576323987538, "story_em": 0.0}
```
