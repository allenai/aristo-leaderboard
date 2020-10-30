import json
import logging
import numpy as np
from typing import List, Dict, Any
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Field
from allennlp.data.fields import ListField, ArrayField, LabelField, MetadataField, SequenceLabelField
from allennlp.data.fields.text_field import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, WordpieceIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp_reasoning_explainqa.common.utils import normalize_text
from collections import Counter
from allennlp_reasoning_explainqa.common.constants import *
import random
from allennlp.data.tokenizers.word_splitter import WordSplitter, BertTokenizer, BertBasicWordSplitter
from typing import List, Optional
import pytorch_pretrained_bert.tokenization

logger = logging.getLogger(__name__)


@WordSplitter.register("bert-basic-custom")
class BertBasicWordSplitterCustom(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """

    def __init__(self,
                 do_lower_case: bool = True,
                 never_split: Optional[List[str]] = None) -> None:
        # super().__init__()
        print("******* never_split = ", never_split)
        never_split = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', 'start0', 'start1']  # ,'[unused0]','[unused1]']
        for i in range(100):
            never_split.append('[unused' + str(i) + ']')
        if never_split is None:
            # Let BertTokenizer use its default
            self.basic_tokenizer = BertTokenizer(do_lower_case)
        else:
            print("******* never_split = ", never_split)
            # 0/0
            self.basic_tokenizer = BertTokenizer(do_lower_case, never_split)
        print("**** self.basic_tokenizer = ", self.basic_tokenizer.never_split)
        print(self.basic_tokenizer.tokenize('checking start0 is start1 is start2 is unused0 is [unused0]'))
        print(self.basic_tokenizer.tokenize('checking start0 is start1 is start2 is unused0 is [unused0]'))
        # 0/0

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(sentence)]

    def tokenize(self, sentence: str) -> List[Token]:
        return self.split_words(sentence)


@TokenIndexer.register("bert-pretrained-custom")
class PretrainedBertIndexerCustom(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.
        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True) -> None:

        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")

        bert_tokenizer = pytorch_pretrained_bert.tokenization.BertTokenizer.from_pretrained(pretrained_model,
                                                                                            do_lower_case=do_lowercase)
        bert_tokenizer.basic_tokenizer.never_split = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "start0", "start1",
                                                      "[unused0]", "[unused1]", "[UNUSED1]", "[UNUSED0]"]
        for i in range(100):
            bert_tokenizer.basic_tokenizer.never_split.append('[unused' + str(i) + ']')
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         namespace="bert",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces,
                         do_lowercase=do_lowercase,
                         never_lowercase=never_lowercase,
                         start_tokens=["[CLS]"],
                         end_tokens=["[SEP]"],
                         separator_token="[SEP]",
                         truncate_long_sequences=truncate_long_sequences)


@DatasetReader.register("qasc_chain_reader")
class QASCChainReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_pieces: int = 512,
                 debug: bool = False,
                 negative_sampling_rate: float = 1.0,
                 negative_sampling_rate_val: float = None,
                 skip_negative_choices: bool = True,
                 chain_type: str = CHAINTYPE_F1_F2_CF,
                 turk_single_fact_label: str = NEGATIVE_LABEL,
                 force_add_gold_chain: bool = True,
                 force_add_gold_chain_val: bool = False,  # new
                 use_tag_representation: bool = False,
                 train_data_fraction: float = None,
                 compute_retrieval_baseline:bool = False,
                 data_version : str = 'qasc',
                 lazy: bool = False) -> None:

        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitterCustom(
            never_split=["start1", "start2"])  # tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_pieces = max_pieces
        self.debug_mode = debug
        self.chain_type = chain_type
        assert chain_type == CHAINTYPE_F1_F2_CF
        self.negative_sampling_rate = negative_sampling_rate
        self.skip_negative_choices = skip_negative_choices

        assert negative_sampling_rate >= 0 and negative_sampling_rate <= 1.0
        if negative_sampling_rate_val is None:
            self.negative_sampling_rate_val = negative_sampling_rate
        else:
            self.negative_sampling_rate_val = negative_sampling_rate_val

        self.force_add_gold_chain = force_add_gold_chain
        if force_add_gold_chain_val is None:
            force_add_gold_chain_val = force_add_gold_chain
        self.force_add_gold_chain_val = force_add_gold_chain_val

        self.turk_label_to_label_mapper = {'yes': POSITIVE_LABEL,
                                           'no': NEGATIVE_LABEL,
                                           'fact1': turk_single_fact_label,
                                           'fact2': turk_single_fact_label,
                                           '?':NEGATIVE_LABEL,
                                           'fact12':NEGATIVE_LABEL,
                                           'not-quite':NEGATIVE_LABEL,
                                           'unsure':NEGATIVE_LABEL,
                                           'bad-answer':NEGATIVE_LABEL}

        self.use_tag_representation = use_tag_representation
        self._non_gold_skip_count = {}
        self.data_version = data_version
        self._instance_num = 0
        self._train_data_fraction = train_data_fraction
        self.compute_retrieval_baseline = compute_retrieval_baseline
        self._se_none_cnt = 0


    def _get_chain(self, s1, s2, s3,
                   id=None, choice_type='gold', split=None,
                   label=None, labeltext=None, chain=None,
                   chain_id=None, unlabelled=None):
        return [s1, s2, s3,
                {'id': id, 'choice_type': choice_type, 'label': label, 'labeltext': labeltext,
                 'chain_id': chain_id, 'unlabelled': unlabelled, 'grc':chain[2]['grc']}]


    @overrides
    def _read(self, file_path: str):
        self._instance_num = 0
        print("Reading instances from file at: ", file_path)
        print("---->>> self.data_version = ", self.data_version)

        if self.data_version == 'obqa':
            from allennlp_reasoning_explainqa.data.analysis.entity_detection import EntityDetectionOverlap
            from allennlp_reasoning_explainqa.data.analysis.grc_transformation import GRCTransform
            self.grc_transform = GRCTransform()
            self.entity_detection_model = EntityDetectionOverlap(OVERLAPPING_ENTITIES_KEY)
            print("---->file_path = ", file_path)
            for i,row in enumerate(open(file_path, 'r').readlines()):
                # QID	Chain#	Tag	Question	Answer	Fact1	Fact2	WOL score	Turk	Turks	Extra Facts DF
                if i==0:
                    continue
                values = row.split('\t')
                txt1 = normalize_text(values[5].strip())
                txt2 = normalize_text(values[6].strip())
                df = normalize_text(values[11].strip())
                qid = normalize_text(values[0].strip())
                cid = values[1].strip()
                turk_label = values[8].strip()
                wol_score = float(values[7].strip())
                chain = [txt1,
                         txt2,
                         df,
                         {'id': qid, 'chain_id': cid,
                          'label': self.turk_label_to_label_mapper[turk_label],
                          'choice_type': CORRECT_OPTION_TAG,
                          'score': float(wol_score)}
                ]
                if self.use_tag_representation:
                    opts = {}
                    vals = {'fact1': chain[0], 'fact2': chain[1], 'cf': chain[2], 'id': qid + '_' + cid, 'option': None,
                            'opts': opts}
                    pattern = self.entity_detection_model.get_pattern(vals)
                    chain[3][OVERLAPPING_ENTITIES_KEY] = {'txt12_candidates': pattern['txt12_candidates'],
                                                               'txt1c_candidates': pattern['txt1c_candidates'],
                                                               'txt2c_candidates': pattern['txt2c_candidates']
                                                               }
                    print("vals = ", json.dumps(vals, indent=4))
                    print("chain[3][OVERLAPPING_ENTITIES_KEY] = ",
                          json.dumps(chain[3][OVERLAPPING_ENTITIES_KEY], indent=4))
                    grc_form = self.grc_transform.get_grc_form([{'text': chain[0]},
                                                                     {'text': chain[1]},
                                                                     {'df': chain[2],
                                                                      OVERLAPPING_ENTITIES_KEY: chain[3][
                                                                          OVERLAPPING_ENTITIES_KEY]},
                                                                     ],
                                                                    debug=False)
                    print("========= grc_form= ", grc_form)
                    chain[0], chain[1], chain[2] = grc_form[0], grc_form[1], grc_form[2]

                self._instance_num += 1
                label = {POSITIVE_LABEL:1, NEGATIVE_LABEL:0}[chain[3]['label']]
                yield self.text_to_instance([chain], [label])
            return

        self._non_gold_skip_count[file_path] = 0
        pos_count = 0
        neg_count = 0
        gold_not_found_count = 0
        gold_found_count = 0
        split = 'train'
        negative_sampling_rate = self.negative_sampling_rate
        force_add_gold_chain = self.force_add_gold_chain
        if file_path.count('dev') > 0:
            split = 'dev'
            negative_sampling_rate = self.negative_sampling_rate_val
            force_add_gold_chain = self.force_add_gold_chain_val
        if file_path.count('test') > 0:
            split = 'test'
            force_add_gold_chain = self.force_add_gold_chain_val
            negative_sampling_rate = self.negative_sampling_rate_val
        print("split = ", split, " || force_add_gold_chain = ", force_add_gold_chain)

        data = []
        with open(file_path) as f:
            for line in f:
                tmp_line = json.loads(line)
                data.append(tmp_line)
        data = data[0]
        print("len of data = ", len(data))
        print("[_read]: ", len(data), data[0].keys(), data[0]['question'].keys(),
              data[0]['question']['choices'][0].keys(),
              data[0]['answerKey'])

        if split == "train":
            if self._train_data_fraction is not None:
                data = random.sample(data, int(self._train_data_fraction * len(data)))

        found = 0
        no_chain = 0
        pos_count_dist = []

        for inst_num, row in enumerate(data):

            skip_negative_choices = self.skip_negative_choices
            negative_sampling_rate_line = negative_sampling_rate
            valid_explanation_chains = []
            negative_chains = []
            correct_choice = row['answerKey']
            gold_is_retrieved = False
            id = row["id"]
            fact1 = normalize_text(row['fact1'])
            fact2 = normalize_text(row['fact2'])
            gold_chain_df = None
            correct_choice_text = None

            for c in row['question']['choices']:

                if c['label'] != correct_choice and skip_negative_choices:
                    continue

                if c['label'] != correct_choice:
                    choice_type = 'incorrect_option'
                else:  # correct choice
                    choice_type = 'correct_option'
                    gold_chain_df = c.get('df', None)
                    correct_choice_text = c['text']

                chains = c["chains"]
                negative_chains_added_for_this_choice = 0

                if len(chains) > 0:

                    for idx, chain in enumerate(chains):

                        if self.debug_mode and idx > 1:
                            break

                        txt1 = normalize_text(chain[0]['text'])
                        txt2 = normalize_text(chain[1]['text'])
                        label = NEGATIVE_LABEL
                        if (txt1 == fact1 and txt2 == fact2) or (txt1 == fact2 and txt2 == fact1):  # gold chain
                            if c['label'] == correct_choice:
                                found += 1
                                if not gold_is_retrieved:
                                    gold_found_count += 1
                                gold_is_retrieved = True
                                label = POSITIVE_LABEL
                        else:
                            label = NEGATIVE_LABEL
                            # if c['label'] == correct_choice:
                            label_turk = chain[2]['turk_label']
                            if label_turk is not None and label_turk['label'] is not None:
                                label = self.turk_label_to_label_mapper[label_turk['label']]

                        s3 = c['df']
                        if s3 is None:
                            self._se_none_cnt += 1
                            s3 = row['question']['stem'] + ' ' + c['text']
                        s3 = normalize_text(s3)

                        chain_to_add = self._get_chain(txt1, txt2, s3, id, choice_type, split,
                                                       c['label'], c['text'], chain,
                                                       chain_id=chain[2].get('chain_id', None))
                        chain_to_add[3]['original'] = txt1 + SEP + txt2 + SEP + s3
                        chain_to_add[3]['label'] = STRLABEL_TO_INT[label]
                        chain_to_add[3]['score'] = chain[0]['score'] + chain[1]['score']
                        chain_to_add[3]['question'] = row['question']['stem']
                        chain_to_add[3]['answer'] = c['text']
                        if label == POSITIVE_LABEL:
                            valid_explanation_chains.append(chain_to_add)  # f1,f2,fcombined
                        else:
                            negative_chains.append(chain_to_add)  # f1,f2,fcombined
                            negative_chains_added_for_this_choice += 1

                else:
                    no_chain += 1

                if c['label'] == correct_choice and force_add_gold_chain and (not gold_is_retrieved):
                    s3 = gold_chain_df
                    if s3 is None:
                        self._se_none_cnt += 1
                        s3 = row['question']['stem'] + ' ' + c['text']
                    s3 = normalize_text(s3)
                    chain_to_add = self._get_chain(fact1, fact2, s3, id, 'gold', split, correct_choice,
                                                   correct_choice_text, chain=[None,None,{'grc':c['grc']}],
                                                   chain_id=row.get('chain_id', None))
                    chain_to_add[3]['original'] = fact1 + SEP + fact2 + SEP + s3
                    chain_to_add[3]['score'] = 0.0 # not retrieved. so assigning default retrieval score of 0
                    chain_to_add[3]['question'] = row['question']['stem']
                    chain_to_add[3]['answer'] = c['text']
                    valid_explanation_chains.append(chain_to_add)

            invalid_explanation_chains = negative_chains
            invalid_explanation_chains = [chain for chain in invalid_explanation_chains if np.random.rand() < negative_sampling_rate_line]
            all_chains = valid_explanation_chains + invalid_explanation_chains
            invalid_chains_labels = [0] * len(invalid_explanation_chains)
            valid_chains_labels = [1] * len(valid_explanation_chains)
            pos_count_dist.append(len(valid_chains_labels))
            pos_count += len(valid_chains_labels)
            neg_count += len(invalid_chains_labels)
            all_labels = np.array(valid_chains_labels + invalid_chains_labels, dtype=np.int)

            for chain, label in zip(all_chains, all_labels):
                if self.use_tag_representation:
                    grc_form = chain[3]['grc']
                    chain[0], chain[1], chain[2] = grc_form[0], grc_form[1], grc_form[2]
                    yield self.text_to_instance([chain], [label])
                else:
                    yield self.text_to_instance([chain], [label])
            if self.debug_mode  and inst_num > 7:
                break

        print(" [read] file_path= ", file_path, "===> gold_not_found_count =", gold_not_found_count)
        print(" [read] file_path= ", file_path, "===> gold_found_count =", gold_found_count)
        print(" [read] file_path= ", file_path, "===> pos_count=", pos_count)
        print(" [read] file_path= ", file_path, "===> pos_count_dist=", Counter(pos_count_dist))
        print(" [split]=  ", split)
        print(" [read] split =  ", split, " || se_none_cnt = ", self._se_none_cnt)
        self._se_none_cnt = 0


    @overrides
    def text_to_instance(self, chains, labels=None, info=None, entity_info=None) -> Instance:

        if self.debug_mode  :
            print("-->> chains = ", chains)
            print("-->> labels = ", labels)
            print("-->> chain3 labels = ", [chain[3]['label'] for chain in chains])

        fields: Dict[str, Field] = {}

        metadata = {
            "id": None,
            "instance_num": self._instance_num,
            "choice_type": None,
            "all_chains": chains,
            "labels": labels,
            "labeltext": None,
            "chain_id": chains[0][3]['chain_id'],
            "info": info,
            "original": chains[0][3].get('original',None)
        }

        self._instance_num += 1
        all_ids = [chain[3]['id'] for chain in chains]
        assert len(all_ids) == 1

        all_chains = [chain[:3] for chain in chains]
        metadata['id'] = all_ids[0]
        metadata['choice_type'] = [chain[3]['choice_type'] for chain in chains][0]
        metadata['label'] = labels[0]  # [ chain[ 3 ][ 'label' ] for chain in chains ][ 0 ]
        metadata['labeltext'] = [chain[3].get('labeltext', None) for chain in chains][0]
        metadata['score'] = [chain[3].get('score',0.0) for chain in chains]

        all_chains = [SEP.join(chain) for chain in all_chains]
        all_chains = [self._tokenizer.tokenize(chain) for chain in all_chains]
        all_chains = [TextField(chain, self._token_indexers) for chain in all_chains]

        all_chains = all_chains[0]
        if labels is not None:
            all_labels = LabelField(str(labels[0]))
        else:
            all_labels = labels
        fields['tokens'] = all_chains
        fields['label'] = all_labels
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields=fields)

