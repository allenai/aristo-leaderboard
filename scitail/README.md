# SciTail

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.

## Example usage

To evaluate dummy predictions (every pair of sentences is predicted to entail) against the SciTail dataset, run this:

```
% python3 evaluator/evaluator.py -a data/test_answers.jsonl -p data/dummy-test-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.39604891815616183}
```

Replace `dummy-test-predictions.csv` with your predictions to compute your test score. 


You can also verify your predictions against the Dev set by running:

```
% python3 evaluator/evaluator.py -a data/dev_answers.jsonl -p <path to your dev predictions> -o metrics.json

```

