from allennlp_reasoning_explainqa.data.dataset_reader.qasc_chains_reader import QASCChainReader
from allennlp_reasoning_explainqa.training.metrics import F1MeasureCustom, ExplanationEval, PrecisionEval, \
    F1MeasureCustomRetrievalEval
from allennlp_reasoning_explainqa.common.constants import *
from allennlp.data.fields.text_field import TextField
from allennlp.data.fields import ListField, ArrayField, LabelField, MetadataField, SequenceLabelField
import sys
from collections import Counter

data_version = sys.argv[1]  #'qasc'
split =  sys.argv[2] #'dev'
thresh = sys.argv[3] # -1

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
        #print("verbose_chain = ", verbose_chain)
        all_chains.append([token_toks,verbose_chain,label])
        all_labels.append(label)
        score = ins['metadata'].metadata['score'][0]
        id = ins['metadata'].metadata['id']
        print("score,label,id = ", score,label,id)
        f1eval(int(label), score)
        explanation_eval(id, CORRECT_OPTION_TAG, int(label), score)
        cnt += 1

    print("Counter(all_labels) = ", Counter(all_labels))
    thresh = float(thresh.strip())
    if thresh < 0:
        print("f1.get_metric() = ", f1eval.get_metric(reset=True))
    else:
        print("f1.get_metric() = ", f1eval.get_metric(reset=True, given_thresh=thresh))
    print("explanation_eval.get_metric() = ", explanation_eval.get_metric(reset=True))


