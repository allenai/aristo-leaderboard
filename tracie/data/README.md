This directory contains the training and test files for evaluating predictions,
and a sample prediction file.

The files `train_iid.jsonl` and `train_uniform.jsonl` are two versions of the
training data that can be used. These two training sets refer to splits, based
on a strict `iid` split and more balanced `uniform` split.

The file `test.jsonl` has the test questions, without labels.

Each example in these files looks like the following:

```json
{
  "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
  "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
  "label": "contradiction"
}
```

and consists of three fields:

* `query` (or hypothesis)
* `story` (or premise)
* `label` (the inference label; this is absent in `test.jsonl`)

The file `train_iid_predictions.jsonl` shows an example prediction file for the `iid`
training split that can be evaluated against `train_iid.jsonl`.
