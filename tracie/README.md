# TRACIE

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* `data_public` publicly available train and test sets (with hidden test labels), along with example prediction files (for testing evaluator).

## Example usage

To evaluate your predictions, run the following (on a toy prediction file that
guesses `entailment` for every train instance, called `train_predictions.txt`). 

```sh
% python3 evaluator/evaluator.py --question_answers data_public/train_iid.jsonl --predictions data_public/train_iid_predictions.txt  --output metrics.json --train_type train_iid

% cat metrics.json
{"train_type": "train_iid", "total_acc": 0.5, "start_acc": 0.5, "end_acc": 0.5, "story_em": 0.0}
```

For usage of the evaluator, see the [evaluator README](evaluator/).
