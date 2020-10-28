This directory contains train and test sets in `train.jsonl` and
`test.jsonl`.

We do not have an explicit `dev` split; we use a portion of the training set to
tune models on `train.jsonl`.

Each line in the file has an example with a JSON structure with three fields,
expanded to multiple lines here for legibility:

```js
{
  "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
  "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
  "gold_label": "contradiction"
}
```

The fields are:

* `story` (the premise)
* `query` (the hypothesis)
* `gold_label` (the inference label); this should be either `entailment` or `contradiction`.

The file `train.jsonl` has 1171 examples, and the file `test.jsonl` has 4251 examples.

In `test.jsonl`, the inference labels are hidden and replaced with `"-"`.

The file `train-dummy-predictions.txt` is a dummy prediction of `entailment`
for all 1171 examples `train.jsonl`, and can be used with the TRACIE evaluator
to validate behavior.

The file `test-dummy-predictions.txt` is a dummy prediction of `entailment` for
all 4251 examples in `test.jsonl`, and can be used with the TRACIE evaluator,
and an unblinded `test.jsonl` file (unpublished) to validate behavior. For
example, you can submit this file to the TRACIE leaderboard to get a successful
evaluation.
