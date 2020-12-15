The mapping of chain ids to correct labels for the dev and test splits are in
these files:

* dev: chainid_to_label_dev.json
* test: chainid_to_label_test.json

## Dummy predictions

As a convenienece for testing the evaluator, two "dummy" prediction files
are provided which give a score of 0.5 to all chains for both splits:

* dev: dummy_predictions_dev.jsonl
* test: dummy_predictions_test.jsonl

These prediction files were created like this:

* dev: `cat chainid_to_label_dev.json  | jq -c '. | keys[] | {"chain_id":., "score":0.5}' > dummy_predictions_dev.jsonl`
* test: `cat chainid_to_label_test.json | jq -c '. | keys[] | {"chain_id":., "score":0.5}' > dummy_predictions_test.jsonl`

You can use these as inputs to the predictor, to confirm that the evaluator is working as expected.

The scores you should expect from these dummy predictions are:

* dev: {"auc_roc": 0.5, "explainP1": 0.21653971708378672, "explainNDCG": 0.476369331557765}
* test: {"auc_roc": 0.5, "explainP1": 0.23497267759562843, "explainNDCG": 0.4873611217679728}
