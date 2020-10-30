from allennlp.models.model import Model
from allennlp_reasoning_explainqa.common.constants import *
from overrides import overrides
import torch
from allennlp_reasoning_explainqa.common.utils import *

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import json

@Predictor.register('interactive')
class NewPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 data_type=None) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
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
        # vals = json.loads(line.strip())
        f1 = input()
        f2 = input()
        df = input()
        # f1 = vals[0]
        # f2 = vals[1]
        # df = vals[2]
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
        outputs["score"] = outputs["probs"][1]
        return json.dumps(outputs) + "\n"

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        chain = [ normalize_text(json_dict['txt1']),
                  normalize_text(json_dict['txt2']),
                  normalize_text(json_dict['df']),
                 {'id': None, 'chain_id': None,
                  'unlabelled': 0,
                  'label': 1,
                  'choice_type': CORRECT_OPTION_TAG}]  # f1,f2,fcombined
        if self._dataset_reader.use_tag_representation:
            opts = {'exclude_verbs': True,
                    'keeponly_start_end_nouns': False,
                    'keeponly_end_nouns': True
                    }
            vals = {'fact1': chain[0], 'fact2': chain[1], 'cf': chain[2], 'id': None, 'option': None, 'opts': opts}
            pattern = self.entity_detection_model.get_pattern(vals)
            chain[3][OVERLAPPING_ENTITIES_KEY] = {'txt12_candidates': pattern['txt12_candidates'],
                                                     'txt1c_candidates': pattern['txt1c_candidates'],
                                                     'txt2c_candidates': pattern['txt2c_candidates']
                                                     }
            # print("vals = ", json.dumps(vals, indent=4))
            print("chain[3][OVERLAPPING_ENTITIES_KEY] = ",
                  json.dumps(chain[3][OVERLAPPING_ENTITIES_KEY], indent=4))
            grc_form = self.grc_transform.get_grc_form( [ {'text':chain[0]},
                                           {'text': chain[1]},
                                           {'df': chain[2],
                                            OVERLAPPING_ENTITIES_KEY:chain[3][OVERLAPPING_ENTITIES_KEY]},
                                           ] ,
                                           debug=False)
            print("========= grc_form: ", grc_form)
            chain[0], chain[1], chain[2] = grc_form[0], grc_form[1], grc_form[2]
        return self._dataset_reader.text_to_instance([chain],[1]) # here 1 is just a dummy label


# MODEL_NAME=grc
# CUDA_VISIBLE_DEVICES=0 python -m allennlp.run predict --output-file predictions/"$MODEL_NAME".interactive --weights-file tmp/"$MODEL_NAME"/best.th --include-package allennlp_reasoning_explainqa  --overrides '{ }' --cuda-device 0 tmp/"$MODEL_NAME"/model.tar.gz ../data/eqasc_perturbed/turk_modified_test.tsv --predictor interactive
