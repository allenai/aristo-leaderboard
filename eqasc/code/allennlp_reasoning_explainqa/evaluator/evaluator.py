from allennlp_reasoning_explainqa.training.metrics.confusion_matrix import *
from allennlp_reasoning_explainqa.training.metrics.explanation_eval import *
from allennlp_reasoning_explainqa.evaluator.constants import *
import sys


def evaluate_eqasc(fname1, split):

    data = open(fname1,'r').readlines()
    assert len(data) == eqasc_test_total_chain_cnt
    data = [json.loads(row) for row in data]
    chainid_to_label = json.load(open('../evaluator_data/eqasc/chainid_to_label_'+split+'.json','r'))
    f1eval = F1MeasureCustomRetrievalEval(pos_label=1, save_fig=False)
    explanation_eval = ExplanationEval()
    chain_ids_covered = []

    cnt = 0
    for row in data:
        assert 'score' in row, "Prediction should contain field score"
        assert 'chain_id' in row, "Prediction should contain field chain_id"
        score = row['score']
        chain_id = row['chain_id']
        qid = chain_id.strip().split('_')[0]
        print("qid,chain_id,score = ", qid,chain_id,score)
        gtlabel = chainid_to_label[chain_id]
        f1eval(int(gtlabel), score)
        explanation_eval(qid, CORRECT_OPTION_TAG, int(gtlabel), score)
        chain_ids_covered.append(chain_id)
        cnt += 1

    assert len(chain_ids_covered) == len(chainid_to_label), "Found {} chains but expected {} chains".format(len(chain_ids_covered), len(chainid_to_label) ) 
    binclf_performance = f1eval.get_metric(reset=True)
    print("f1.get_metric() = ", binclf_performance )
    explanation_performance =  explanation_eval.get_metric(reset=True)
    print("explanation_eval.get_metric() = ", explanation_performance )
    final_metrics = {
            'auc_roc':binclf_performance['auc_roc'],
            'explainP1':explanation_performance['explainP1'],
            'explainNDCG':explanation_performance['explainNDCG']
            }
    print("="*32)
    print(": auc_roc = ", binclf_performance['auc_roc'])
    print(": P1 = ", explanation_performance['explainP1'])
    print(": explainNDCG = ", explanation_performance['explainNDCG'])
    print("="*32)
    return final_metrics

if __name__ == '__main__':
    fname = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'eqasc_test':
        final_metrics = evaluate_eqasc(fname, split='test')
    elif mode == 'eqasc_dev':
        final_metrics = evaluate_eqasc(fname, split='dev')
    else:
        raise NotImplementedError
    json.dump(final_metrics,open('metrics.json','w'))

# env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/evaluator.py predictions/grc.test.predict eqasc_test
# env PYTHONPATH=. python allennlp_reasoning_explainqa/evaluator/evaluator.py predictions/grc.dev.predict eqasc_dev
