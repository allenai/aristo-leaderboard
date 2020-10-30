## eQASC Evaluator

This script evaluates predictions for eQASC predictions against ground truth annotations and produces metrics.

## Example

```bash
% PREDICTION_FILE_PATH=predictions/grc.test.predict
% EVAL_MODE=eqasc_test
% env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/evaluator.py $PREDICTION_FILE_PATH $EVAL_MODE

% cat metrics.json
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}
```

## Usage

The script takes one input file and produces one output file. It uses data from ../evaluator_data/

Evaluator data consists of chain ids as keys with corresponding labels a values. 
For example:

```bash
% cat ../evaluator_data/eqasc/chainid_to_label_test.json
 {"3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_7": 0, 
 "3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_8": 0, 
 "3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_6": 1, 
...
```


### Input predictions

A predictions file that has predictions in jsonl format. For example:

```bash
%  cat predictions/grc.test.predict | head -n 4
{"score": 0.2023383378982544, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_1"}
{"score": 0.5158032774925232, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_2"}
{"score": 0.17925743758678436, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_5"}
{"score": 0.8793290853500366, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_7"}
```

- Prediction file must have the exact same set of `chain_id` as in the `../evaluator_data/eqasc/chainid_to_label_test.json`


### Output metrics

A JSON file that has three key-value pairs. The keys are as follows: 1) auc_roc 2) explainP1 3)explainNDCG. 
For example:
```bash
% cat metrics.json 
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}
```

### Environment

environment.yml file is provided

## Running in Docker

The eQASC evaluator has many dependencies, like AllenNLP, so if you only want
to run the evaluator on a prediction file, this is the easiest way to do so,
without setting up a local development environment (Conda) with those
dependencies installed.

First, build an image with the evaluator:

```
docker build -t eqasc-evaluator .
```

Then run it with the above files like this:

```
docker run \
  -e PYTHONPATH=. \
  -e PYTHONUNBUFFERED=yes \
  -v $PWD/predictions:/predictions:ro \
  -v $PWD/../evaluator_data/eqasc:/labels:ro \
  -v /tmp:/output:rw \
  --entrypoint python \
  eqasc-evaluator \
  allennlp_reasoning_explainqa/evaluator/evaluator.py \
  /predictions/grc.test.predict \
  /labels/chainid_to_label_test.json \
  /output/metrics.json
```

This evaluates the file `predictions/grc.test.predict` against the labels in
`../evaluator_data/eqasc/chainid_to_label_test.json`, and writes the file
`/tmp/metrics.json` locally:

```
% cat /metrics.json
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}%   
```
