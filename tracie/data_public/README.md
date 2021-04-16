This directory contains the files `train_iid.jsonl`, `train_uniform.json`, and `test.jsonl` (the different training sets refer to different splits based on a strict `iid` split and more balanced `uniform` split).

Each example in the file looks like the following:
```
{
    "query": "event: Tom's teeth are crooked ends before he has braces on for a while",
    "story": "Tom needed to get braces. He was afraid of them. The dentist assured him everything would be fine. Tom had them on for a while. Once removed he felt it was worth it.",
    "gold_label": "contradiction"}
```
and consists of three fields: `story` (or the premise); `query` (or hypothesis) and `gold_label` (the inference label). The testing labels are hidden and replaced with `"-"`

`train_iid_predictions.txt` shows an example output file for the `iid` training split that can be used with the evaluator. 
