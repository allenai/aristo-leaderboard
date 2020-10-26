# TRACIE

* [evaluator](evaluator/) is the program used by the AI2 Leaderboard to evaluate submitted predictions.
* `data_public` publicly available train and test sets (with hidden test labels), along with example prediction files (for testing evaluator).

## Example usage

To evaluate your predictions, run the following (on a toy prediction file that guesses `entailment` for every train instance, called `train_predictions.txt`). 
```

%python3 evaluator/evaluator.py --question_answers data_public/train.jsonl --predictions data_public/train_predictions.txt --output metrics.json

% cat metrics.json
{"total_acc": 0.5012809564474808, "start_acc": 0.500945179584121, "end_acc": 0.5015576323987538, "story_em": 0.0}
```

For usage of the evaluator, see the [evaluator README](evaluator/).
