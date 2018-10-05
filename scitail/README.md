# SciTail

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.

## Example usage

To evaluate dummy predictions (every pair of sentences is predicted to entail) against the SciTail dataset, run this:

```
% python3 evaluator/evaluator.py -a data/answers.jsonl -p data/dummy-predictions.csv -o metrics.json 

% cat metrics.json
{"accuracy": 0.39604891815616183}
```
