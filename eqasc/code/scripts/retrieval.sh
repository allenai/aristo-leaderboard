#!/usr/bin/env bash
mkdir tmp/retrieval
env PYTHONPATH=. python allennlp_reasoning_explainqa/model/retrieval.py qasc dev -1 > tmp/retrieval/retrieval.eqasc.dev.log
env PYTHONPATH=. python allennlp_reasoning_explainqa/model/retrieval.py qasc test 0.35392 > tmp/retrieval/retrieval.eqasc.test.log
env PYTHONPATH=. python allennlp_reasoning_explainqa/model/retrieval.py qasc train 0.35392 > tmp/retrieval/retrieval.eqasc.train.log
env PYTHONPATH=. python allennlp_reasoning_explainqa/model/retrieval.py obqa test 0.35392 > tmp/retrieval/retrieval.eobqa.test.log
