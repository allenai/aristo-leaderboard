This directory contains the files `train_iid.jsonl`, `train_uniform.jsonl` (two
versions of the training data that can be used), and `test.jsonl` (the
different training sets refer to different splits based on a strict `iid` split
and more balanced `uniform` split).

Each example in these files looks like the following:

```json
{
  "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
  "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
  "gold_label": "contradiction"
}
```

and consists of three fields:

* `query` (or hypothesis)
* `story` (or premise)
* `gold_label` (the inference label)

The gold labels in `test.jsonl` are hidden and replaced with `"-"`.

The file `train_iid_predictions.txt` shows an example output file for the `iid`
training split that can be used with the evaluator. 
