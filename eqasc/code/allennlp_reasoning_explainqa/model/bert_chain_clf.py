from typing import Dict, Union, List, Any, Tuple
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp_reasoning_explainqa.common.constants import *
from allennlp_reasoning_explainqa.training.metrics import F1MeasureCustom, ExplanationEval, PrecisionEval
from overrides import overrides
import torch
import pickle
import torch.nn as nn
# from src.utils.allenai.tensor_utils import get_text_field_mask


@Model.register("bert_chain_clf")
class BertChainClassifier(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    use_ig: ``bool``, optional(default: False)
        If True, will incorporate IG features
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    latent clusters
        (2) Mark the closest cluster. update the marked cluster mean using moving average. add a loss term to data representation to come closer to the mean
        When predicting negative, move away from all negative chains ?
        (1) May be first visualize CLS pooled representation on which the classifier operates
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: str,
                 dropout: float = 0.0,
                 num_labels: int = 2,
                 index: str = "bert",
                 trainable: bool = True,
                 negative_class_weight: float = 1.0,
                 classifier_type:str = 'default',
                 pred_file_name: str = None,
                 pred_thresh: float = None,
                 f1_given_thresh: float = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.bert_model.requires_grad = trainable
        in_features = self.bert_model.config.hidden_size
        out_features = num_labels
        self._dropout = torch.nn.Dropout(p=dropout)
        self.pred_file_name = pred_file_name
        self.pred_thresh = pred_thresh
        if classifier_type == "default":
            self._classification_layer = torch.nn.Linear(in_features, out_features)
        else:
            h = int(in_features/4)
            layer1 = torch.nn.Linear(in_features, h)
            layer2 = torch.nn.Linear(h, out_features)
            self._classification_layer = nn.Sequential(
                layer1,
                nn.ReLU(),
                layer2 )
            print("self._classification_layer = ", self._classification_layer)
        self._accuracy = CategoricalAccuracy()
        self._f1 = F1MeasureCustom()
        self.f1_given_thresh = f1_given_thresh
        self._class_weights = torch.Tensor([negative_class_weight, 1.0])
        self._loss = torch.nn.CrossEntropyLoss(self._class_weights)
        self._index = index
        print("**** in_features = ", in_features)
        print(" **** self.bert_model.config = ", self.bert_model.config)
        self._explanation_eval = ExplanationEval(pos_label=1, neg_label=0)
        # self._precision_eval = PrecisionEval()
        initializer(self._classification_layer)


    def forward(self,  # type: ignore
                tokens = None,
                label: torch.LongTensor = None,
                label_option_type: torch.LongTensor = None,
                segment_ids = None,
                metadata: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata: Metadata from the dataset, optional
            From a ``MetaDataField``
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """


        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        vals = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        attn_list, pooled = vals   # return encoded_layers, pooled_output

        pooled = self._dropout(pooled)

        # apply classification layer
        logits = self._classification_layer(pooled)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits,
                       "probs": probs,
                       "attention": attn_list}

        output_dict["all_chains"] = [metadata_i["all_chains"] for metadata_i in metadata]
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            output_dict["gold_label"] = label.data.cpu().numpy()
            output_dict["id"] = [metadata_i["id"] for metadata_i in metadata]
            output_dict["choice_type"] = [metadata_i["choice_type"] for metadata_i in metadata]
            output_dict["label"] = [metadata_i["label"] for metadata_i in metadata]
            output_dict["labeltext"] = [metadata_i["labeltext"] for metadata_i in metadata]
            output_dict["instance_num"] = [metadata_i["instance_num"] for metadata_i in metadata]
            output_dict["tokens_from_vocab"] = [ [self.vocab._index_to_token['bert'].get(idx.cpu().data.item(), '.')
                                                  for idx in batch_indices ]
                                                 for batch_indices in input_ids ]

            self._accuracy(logits, label)
            self._f1(logits, label, probs)

            probs_numpy = output_dict['probs'].data.cpu().numpy()
            for i,metadata_i in enumerate(metadata):
                ques_id = metadata_i['id']
                ground_truth_label = label[i].data.cpu().item()
                choice_type = metadata_i['choice_type']
                score = probs_numpy[i][1] # score of positive class
                self._explanation_eval(ques_id, choice_type, ground_truth_label, score)
                # self._precision_eval(ques_id, choice_type, ground_truth_label, score)

            predictions = output_dict["probs"]
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
            output_dict['pred_labels'] = []
            for idx,prediction in enumerate(predictions_list):
                pred_label_idx = prediction.argmax(dim=-1).item()
                output_dict[ 'pred_labels' ].append(pred_label_idx)

        return output_dict



    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """

        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_token_from_index(label_idx, namespace="labels")
            classes.append(label_str)
        output_dict["pred_label"] = classes
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_values = self._f1.get_metric(reset, given_thresh=self.f1_given_thresh) #f1,prec,rec
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        metrics.update(f1_values)
        explain_evals = self._explanation_eval.get_metric(reset)
        # qa_evals = self._precision_eval.get_metric(reset)
        metrics.update(explain_evals)
        # metrics.update(qa_evals)
        metrics.update(f1_values)
        return metrics



from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('binclf_predictor')
class BinaryClassifierPredictor(Predictor):

    def __init__(self, model: Model,
                 dataset_reader: DatasetReader,
                 chain_type:str = CHAINTYPE_F1_F2_CF) -> None:
        super().__init__(model, dataset_reader)
        self.chain_type = chain_type
        self._fname_to_dump = model.pred_file_name
        print("model.pred_file_name = ", model.pred_file_name)
        fw = open(self._fname_to_dump, "w")
        vals = ['id',
                'chain_id',
                'question',
                'answer',
                'chain[0]',
                'chain[1]',
                'chain[2]',
                'chain-original',
                'pred_probs[0]',
                'pred_probs[1]',
                'gt-label',
                'pred-label'
                ]
        fw.write('\t'.join(vals))
        fw.write('\n')
        fw.close()
        self._total_cnt = 0
        self.thresh = model.pred_thresh


    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        self._total_cnt += 1
        print("outputs: ", outputs)
        if outputs["probs"][1]>self.thresh:
            pred_label = 1
        else:
            pred_label = 0
        fw = open(self._fname_to_dump, "a")
        vals = [outputs["id"],
                outputs["all_chains"][0][3]['chain_id'],
                outputs["all_chains"][0][3]['question'],
                outputs["all_chains"][0][3]['answer'],
                outputs["all_chains"][0][0],
                 outputs["all_chains"][0][1],
                 outputs["all_chains"][0][2],
                 outputs["all_chains"][0][3]['original'],
                 str(outputs["probs"][0]),
                 str(outputs["probs"][1]),
                 str(outputs['label']),
                 str(pred_label)
                 ]
        fw.write('\t'.join(vals))
        fw.write('\n')
        fw.close()
        final_outputs = {}
        final_outputs["score"] = outputs["probs"][1]
        final_outputs['chain_id'] = outputs["all_chains"][0][3]['chain_id']
        return json.dumps(final_outputs) + "\n"



import json
import string

def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.strip()


@Predictor.register('new_predictor')
class NewPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 fname_to_dump="pred_dumps.tsv",
                 data_type=None) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
        self._fname_to_dump = fname_to_dump
        fw = open(fname_to_dump, "w")
        fw.close()
        self._data_type = data_type
        self._pred_label_dist = {1:0,0:0}
        self._total_cnt = 0
        if torch.cuda.is_available():
            self._model.cuda()
        if self._dataset_reader.use_tag_representation:
            from allennlp_reasoning_explainqa.data.analysis.entity_detection import EntityDetectionOverlap
            self.entity_detection_model = EntityDetectionOverlap(OVERLAPPING_ENTITIES_KEY)
        from allennlp_reasoning_explainqa.data.analysis.grc_transformation import GRCTransform
        self.grc_transform = GRCTransform()
        self.thresh = model.pred_thresh


    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        if self._data_type == 'replace_test_new_mod':
            # QID	Fact1	Fact2	Combined	Fact1Edited	Fact2Edited	CombinedEdited	Fact1Change	Fact2Change	CombinedChange
            vals = line.strip().split('\t')
            f1 = vals[4]
            f2 = vals[5]
            df = vals[6]
            line = {'df': normalize_text(df),
                    'txt1': normalize_text(f1),
                    'txt2': normalize_text(f2)}
            print("===>>>>>>> line = ", line)
            return line  # json.loads(line)
        elif self._data_type == 'replace_test_new':
            # QID	Fact1	Fact2	Combined	Fact1Edited	Fact2Edited	CombinedEdited	Fact1Change	Fact2Change	CombinedChange
            vals = line.strip().split('\t')
            f1 = vals[1]
            f2 = vals[2]
            df = vals[3]
            line = {'df': normalize_text(df),
                    'txt1': normalize_text(f1),
                    'txt2': normalize_text(f2)}
            print("===>>>>>>> line = ", line)
            return line  # json.loads(line)
        else:
            vals = json.loads(line.strip())
            f1 = vals[0]
            f2 = vals[1]
            df = vals[2]
            line = {'df': normalize_text(df),
                    'txt1': normalize_text(f1),
                    'txt2': normalize_text(f2)}
            print("===>>>>>>> line = ", line)
            return line  # json.loads(line)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        self._total_cnt += 1
        fw = open(self._fname_to_dump, "a")
        vals = [ outputs["all_chains"][0][0],
                 outputs["all_chains"][0][1],
                 outputs["all_chains"][0][2],
                 str(outputs["probs"][0]),
                 str(outputs["probs"][1]),
                 ]
        fw.write('\t'.join(vals))
        fw.write('\n')
        fw.close()
        if outputs["probs"][1]>self.thresh:
            pred_label = 1
        else:
            pred_label = 0
        self._pred_label_dist[pred_label]+=1
        print("self._pred_label_dist = ", self._pred_label_dist)
        outputs["score"] = outputs["probs"][1]
        final_outputs = { 'chain_id':outputs['chain_id'], 'score':outputs['score']  }
        #return json.dumps(outputs) + "\n"
        return json.dumps(final_outputs) + "\n"

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        if self._data_type == 'replace_test' or self._data_type == 'replace_test_new' or self._data_type == 'replace_test_new_mod':
            chain = [ normalize_text(json_dict['txt1']),
                      normalize_text(json_dict['txt2']),
                      normalize_text(json_dict['df']),
                     {'id': None, 'chain_id': None,
                      'unlabelled': 0,
                      'label': 1,
                      'choice_type': CORRECT_OPTION_TAG}]  # f1,f2,fcombined
            if self._dataset_reader.use_tag_representation:
                opts = {}
                vals = {'fact1': chain[0], 'fact2': chain[1], 'cf': chain[2], 'id': None, 'option': None, 'opts': opts}
                pattern = self.entity_detection_model.get_pattern(vals)
                chain[3][OVERLAPPING_ENTITIES_KEY] = {'txt12_candidates': pattern['txt12_candidates'],
                                                         'txt1c_candidates': pattern['txt1c_candidates'],
                                                         'txt2c_candidates': pattern['txt2c_candidates']
                                                         }
                print("vals = ", json.dumps(vals, indent=4))
                print("chain[3][OVERLAPPING_ENTITIES_KEY] = ",
                      json.dumps(chain[3][OVERLAPPING_ENTITIES_KEY], indent=4))
                grc_form = self.grc_transform.get_grc_form( [ {'text':chain[0]},
                                               {'text': chain[1]},
                                               {'df': chain[2],
                                                OVERLAPPING_ENTITIES_KEY:chain[3][OVERLAPPING_ENTITIES_KEY]},
                                               ] ,
                                               debug=False)
                print("========= after  dual: ", grc_form)
                chain[0], chain[1], chain[2] = grc_form[0], grc_form[1], grc_form[2]
            return self._dataset_reader.text_to_instance([chain],[1]) # here 1 is just a dummy label


@Predictor.register('new_predictor_replace_newtest_orig')
class NewPredictorReplaceNewtestOrig(NewPredictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 fname_to_dump="predictions/new_predictor_replace_new_orig.tsv") -> None:
        super().__init__(model, dataset_reader, fname_to_dump, data_type='replace_test_new')


@Predictor.register('new_predictor_replace_newtest_mod')
class NewPredictorReplaceNewtestMod(NewPredictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 fname_to_dump="predictions/new_predictor_replace_new_mod.tsv") -> None:
        super().__init__(model, dataset_reader, fname_to_dump, data_type='replace_test_new_mod')
