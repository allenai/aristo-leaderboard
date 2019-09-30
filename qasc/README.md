# QASC

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* `data` have example prediction files

## Example usage

To evaluate dummy predictions (every question is predicted to be `A`) against the train dataset, run this:

```
% python3 evaluator/evaluator.py -qa data/train.jsonl -p data/train-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.12417014998770592}
```
