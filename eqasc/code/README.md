## eQASC Evaluator

This script evaluates predictions for eQASC predictions against ground truth annotations and produces metrics.

Hint: If you are in a hurry and want to simply evaluate your predictions, run the evaluator in Docker.

## Usage

The program [evaluator.py](allennlp_reasoning_explainqa/evaluator/evaluator.py) takes three arguments:

1. The filename of a prediction file.
2. The filename of the labels to evaluate against.
3. The filename where metrics will be written.

### Prediction file

The predictions file should hold multiple JSON objects, with each object having a score, and a chain ID. For example:

```bash
%  cat predictions/grc.test.predict | head -n 4
{"score": 0.2023383378982544, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_1"}
{"score": 0.5158032774925232, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_2"}
{"score": 0.17925743758678436, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_5"}
{"score": 0.8793290853500366, "chain_id": "3C44YUNSI1OBFBB8D36GODNOZN9DPA_1_7"}
```

The chain IDs must match those in the labels. (See below.)

The file `predictions/grc.test.predict` in this repo contains an example
prediction for the test labels. It was made by doing ... TODO

### Labels file

The labels file holds a single JSON object with keys being chain IDs and values
being labels. It looks like this:

```
% cat ../evaluator_data/eqasc/chainid_to_label_test.json
 {"3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_7": 0, 
 "3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_8": 0, 
 "3GM6G9ZBKNWCBXAS7DE3CDBF13STML_1_6": 1, 
...
```

The file `../evaluator_data/eqasc/chainid_to_label_test.json` in this repo
contains the labels for test chains.

### Output metrics

A "metrics" file will be written that contains evaluation scores.

This file holds a single JSON structure with three key-value pairs. The keys are:

* `auc_roc` -- This is .. TODO
* `explainP1` -- This is .. TODO
* `explainNDCG` - This is ... TODO

Example:

```bash
% cat metrics.json 
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}
```

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
% cat /tmp/metrics.json
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}%   
```

See below for an explanation of the three arguments to the `evaluator.py` script.

## Running locally

You'll have to install dependencies with Conda, following the environment.yml file.

After you've done that, run the evaluator like this:

```bash
% env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/evaluator.py predictions/grc.test.predict ../evaluator_data/eqasc/labels/chainid_to_label_test.json /tmp/metrics.json
```

This evaluates the file `predictions/grc.test.predict` against the labels in
`../evaluator_data/eqasc/chainid_to_label_test.json`, and writes the file
`/tmp/metrics.json` locally:

```
% cat /tmp/metrics.json
{"auc_roc": 0.8457533894216488, "explainP1": 0.5387978142076503, "explainNDCG": 0.6376201537170901}
```

## Testing

The script `test-with-docker.sh` uses the Docker method to exercise the
evaluator and confirm expected scores.
