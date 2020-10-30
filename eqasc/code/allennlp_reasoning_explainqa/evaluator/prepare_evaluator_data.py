from allennlp_reasoning_explainqa.data.dataset_reader.qasc_chains_reader import QASCChainReader
from allennlp_reasoning_explainqa.training.metrics import F1MeasureCustom, ExplanationEval, PrecisionEval, \
    F1MeasureCustomRetrievalEval
from allennlp_reasoning_explainqa.common.constants import *
from allennlp.data.fields.text_field import TextField
from allennlp.data.fields import ListField, ArrayField, LabelField, MetadataField, SequenceLabelField
import sys
from collections import Counter
import json

data_version = sys.argv[1]  #'qasc'
split =  sys.argv[2] #'dev'
fwname = sys.argv[3] #

if __name__ == "__main__":

    reader = QASCChainReader(tokenizer=None,
                             debug=False,
                             skip_negative_choices=True,
                             chain_type=CHAINTYPE_F1_F2_CF,
                             force_add_gold_chain=False,  # False
                             force_add_gold_chain_val=False,  # False
                             use_tag_representation=False,  # True
                             data_version=data_version
                             )


    if data_version=='obqa':
        assert split == 'test'
        fname = '../data/eobqa/obqa_chains.tsv.processed.tsv'
    else:
        fname = "../data/eqasc/eqasc_"+ split + "_grc.json"
    chainid_to_label = {}

    f1eval = F1MeasureCustomRetrievalEval(pos_label=1, save_fig=False)
    explanation_eval = ExplanationEval()
    cnt = 0
    all_chains = []
    all_labels = []
    for ins in reader._read(fname):
        print("ins = ", ins)
        print("cnt = ", cnt)
        tokens:TextField = ins['tokens']
        token_toks = tokens.tokens
        token_toks = [tok.text for tok in token_toks]
        lab:LabelField = ins['label']
        label = lab.label
        verbose_chain = ins['metadata'].metadata['original']
        all_chains.append([token_toks,verbose_chain,label])
        all_labels.append(label)
        score = ins['metadata'].metadata['score'][0]
        id = ins['metadata'].metadata['id']
        chain_id = ins['metadata'].metadata['chain_id']
        chainid_to_label[chain_id] = int(label)
        cnt += 1

    print("Counter(all_labels) = ", Counter(all_labels))
    json.dump(chainid_to_label, open(fwname,'w'))
    #
    # mkdir ../evaluator_data/
    # mkdir ../evaluator_data/eqasc/
    # env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/prepare_evaluator_data.py qasc dev ../evaluator_data/eqasc/chainid_to_label_dev.json
    # env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/prepare_evaluator_data.py qasc test ../evaluator_data/eqasc/chainid_to_label_test.json
