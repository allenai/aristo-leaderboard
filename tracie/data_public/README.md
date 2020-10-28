This directory contains the files `train.jsonl`, `dev.jsonl` and `test.jsonl`. Please note that `train.jsonl` and `dev.jsonl` are the same file, since we do not have an explicit `dev` split (for our experiments, we use a portion of the training set to tune models on `train.jsonl`).

Each example in the file looks like the following:
```
{
    "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
    "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
    "gold_label": "contradiction"}
```
and consists of three fields: `story` (the premise); `query` (the hypothesis) and `gold_label` (the inference label). The testing labels are hidden and replaced with `"-"`

`train_predictions.txt` shows an example output file that can be used with the evaluator. 
