import json
import os
from overrides import overrides
from typing import Dict, List, Set, Tuple, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()
from allennlp_reasoning_explainqa.common.constants import OVERLAPPING_ENTITIES_KEY

import string
def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

none_ctr = 0


class EntityDetection:

    def __init__(self, etype:str = 'default'):
        self.etype = etype


    def load_data(self, data_dir:str, fname:str):
        data = []
        with open(data_dir + fname) as f:
            for line in f:
                data.append(json.loads(line))
        return data


    def load_data_dct(self, data_dir:str, fname:str):
        data = {}
        with open(data_dir + fname) as f:
            for line in f:
                row = json.loads(line)
                data[row['id']] = row
        return data


    def get_pattern(self, vals: Dict) -> Dict:
        return {'f1': {'text': vals['fact1']}, 'f2':{'text': vals['fact2']} }

    def _dump_call(self, info):
        pass

    def _update_chain(self, chain, pattern):
        chain[0]['pattern'] = pattern['f1']
        chain[1]['pattern'] = pattern['f2']

    def run_for_all_chains(self, dir_path:str, split:str, opts:Dict, debug:bool = False):
        compute_for_gold_only = opts.get('compute_for_gold_only', False)
        use_cf:bool = opts.get('use_cf', True)
        if use_cf:
            assert compute_for_gold_only
        data = self.load_data(data_dir=dir_path, fname=split + '_df.txt')
        target_dir = dir_path + 'analysis/'
        if not os.path.exists(target_dir):
            print("CREATING ", target_dir)
            os.makedirs(target_dir)
        target_path = target_dir + self.etype + split
        ctr = 0
        print("data: = ", len(data))
        assert len(data)==1
        data = data[0]
        # data = json.loads(data[0])
        for jj,row in enumerate(data):
            # print("row = ", row)
            # print("row.keys() = ", row.keys())
            fact1 = normalize_text(row['fact1'])
            fact2 = normalize_text(row['fact2'])
            cf = row['combinedfact']
            ques = row['question']['stem']
            anskey = row['answerKey']
            ans = None
            for c in row['question']['choices']:
                chains = c["chains"]
                #print("c.keys: ", c.keys())
                label = c['label']
                if label == anskey:
                    ans = c['text']
                label_text = c['text']
                if not compute_for_gold_only:
                    for idx, chain in enumerate(chains):
                        txt1 = normalize_text(chain[0]['text'])
                        txt2 = normalize_text(chain[1]['text'])
                        inp = {'fact1': txt1, 'fact2': txt2, 'ques': ques, 'option': label_text, 'cf': cf, 'opts':opts, 'id':row['id']}
                        pattern = self.get_pattern(inp)
                        self._update_chain(chain, pattern)
                        if opts.get('dump_simultaneously',False):
                            self._dump_call({'inp':inp,'pattern':pattern,'opts':opts, 'row':row})
            inp = {'fact1': fact1, 'fact2': fact2, 'ques': ques, 'option': ans, 'cf': cf, 'opts':opts, 'id':row['id']}
            pattern = self.get_pattern(inp)
            row['gold_pattern'] = [{},{}]
            self._update_chain(row['gold_pattern'], pattern)
            if opts.get('dump_simultaneously', False):
                self._dump_call({'inp': inp, 'pattern': pattern, 'opts': opts, 'row':row})
            ctr+=1
            if debug and ctr>20:
                break
            print("row = ", json.dumps(row, indent=4))
            print("data[jj] = ", json.dumps(data[jj], indent=4))
        fwname = target_path+".json"
        print("------- Dumping to ", fwname)
        json.dump(data, open(fwname,'w'))




class EntityDetectionOverlap(EntityDetection):

    def __init__(self, etype:str='default_overlap'):
        super(EntityDetectionOverlap, self).__init__(etype)
        self.dump_fw = None

    def _get_overlaps(self, t1: List[str], t2:List[str]):
        n = len(t1)
        m = len(t2)
        ret = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if t1[i] == t2[j]:
                    ret[i][j] = 1
        return ret

    def _aligns(self, lst1: List[str], lst2: List[str], do_stem:bool = True, restrict_to_nouns: bool = False) -> bool:
        if do_stem:
            lst1 = [v for v in map(stemmer.stem,lst1)]
            lst2 = [v for v in map(stemmer.stem, lst2)]
        if restrict_to_nouns:
            pass
        return set(lst1) == set(lst2)

    def _has_info_words(self, lst: List[str]) -> bool:
        if len(lst)>0 and len([w for w in lst if w not in stopwords_set])>0:
            return True
        return False

    def _prune_subsets(self, set_of_vals:Set[str]) -> Set[str]:
        ret = set([])
        for i,val in enumerate(set_of_vals):
            found = False
            for j,val2 in enumerate(set_of_vals):
                if val!=val2 and val2.count(val) > 0:
                    found = True
                    break
            if not found:
                ret.add(val)
        return ret

    @overrides
    def _dump_call(self, info):
        pass

    @overrides
    def _update_chain(self, chain, pattern):
        chain[0]['pattern'] = chain[1]['pattern'] = pattern

    @overrides
    def get_pattern(self, vals: Dict, print_details=False) -> Dict:
        global none_ctr
        txt1 = normalize_text(vals['fact1'])
        txt2 = normalize_text(vals['fact2'])
        opts = vals['opts']
        use_cf = opts.get('use_cf',True)
        if use_cf:
            print("====>>>> cf = ", vals['cf'])
            if vals['cf'] is None:
                cf = normalize_text(vals['ques'] + ' ' + vals['option'])
                none_ctr += 1
            else:
                cf= normalize_text(vals['cf'])
        else:
            # cf= normalize_text(vals['ques'] + ' ' + vals['option'])
            raise NotImplementedError
        txt12_candidates = self.get_common_entities(txt1, txt2, opts, print_details=print_details)
        txt1c_candidates = self.get_common_entities(txt1, cf, opts, print_details=print_details)
        txt2c_candidates = self.get_common_entities(txt2, cf, opts, print_details=print_details)

        txt12_candidates = {'txt1_candidates': list(txt12_candidates[ 0 ]),
                            'txt2_candidates': list(txt12_candidates[ 1 ]) }
        txt1c_candidates = {'txt1_candidates': list(txt1c_candidates[ 0 ]),
                            'txtc_candidates': list(txt1c_candidates[ 1 ])}
        txt2c_candidates = {'txt2_candidates': list(txt2c_candidates[ 0 ]),
                            'txtc_candidates': list(txt2c_candidates[ 1 ])}
        return dict(txt12_candidates=txt12_candidates,
                    txt1c_candidates=txt1c_candidates,
                    txt2c_candidates=txt2c_candidates)

    def _get_tags(self, text:List[str]) -> List[Tuple[str,str]]:
        ret = nltk.pos_tag(text)
        # print("tagging: ", ret)
        return ret

    def _no_verb(self, tags:List[str]) -> bool:
        for tag in tags:
            if tag.lower().count('vb')>0:
                # print(" found ", tag)
                return False
        return True

    def _starts_ends_noun(self, tags: List[str], end_only:bool =False) -> bool:
        # print( " --- tags = ", tags)
        if len(tags)>0 and tags[-1].lower().count('nn')>0 and (tags[0].lower().count('nn') or end_only )>0:
            return True
        return False

    def get_common_entities(self, txt1:str, txt2:str, opts:Dict, print_details=False) -> Tuple[Set,Set]:
        txt1_tokens = txt1.strip().split()
        txt2_tokens = txt2.strip().split()
        txt1_tags = self._get_tags(txt1_tokens)
        if print_details:
            print("txt1_tags = ", txt1_tags)
        txt1_tokens = [tw[0] for tw in txt1_tags]
        txt1_tags = [tw[1] for tw in txt1_tags]
        txt2_tags = self._get_tags(txt2_tokens)
        if print_details:
            print("txt2_tags = ", txt2_tags)
        txt2_tokens = [tw[0] for tw in txt2_tags]
        txt2_tags = [tw[1] for tw in txt2_tags]
        # ret = self._get_overlaps(txt1_tokens, txt2_tokens)
        txt1_candidates, txt2_candidates = set(), set()
        maxj = 0
        for i in range(len(txt1_tokens)):
            for j in range(len(txt1_tokens), max(i, maxj), -1):
                found = False
                for p in range(len(txt2_tokens)):
                    for q in range(p + 1, len(txt2_tokens)+1):
                        # print("pre testing ", txt1_tokens[i:j], " || ", txt2_tokens[p:q])
                        if not found and self._aligns(txt1_tokens[i:j], txt2_tokens[p:q]):
                            # print("testing ", txt1_tokens[i:j])
                            if self._has_info_words(txt1_tokens[i:j]):
                                if self._no_verb(txt1_tags[i:j]):
                                    if self._starts_ends_noun(txt1_tags[i:j], end_only=True):
                                        txt1_candidates.add(' '.join(txt1_tokens[i:j]))
                                        txt2_candidates.add(' '.join(txt2_tokens[p:q]))
                                        found = True
                                        maxj = max(j, maxj)
                if found:
                    break
        if print_details:
            print("txt1_candidates = ", txt1_candidates)
            print("txt2_candidates = ", txt2_candidates)
        txt1_candidates = self._prune_subsets(txt1_candidates)
        txt2_candidates = self._prune_subsets(txt2_candidates)
        if print_details:
            print("txt1_candidates after pruning = ", txt1_candidates)
            print("txt2_candidates after pruning = ", txt2_candidates)
        return txt1_candidates, txt2_candidates



class ExtractOverlappingConcepts:

    def __init__(self, entity_detection_model_type=OVERLAPPING_ENTITIES_KEY):
        self.entity_detection_model_type = entity_detection_model_type
        # if self.entity_detection_model_type == "openie":
        #     self.entity_detection_model = EntityDetectionAll(etype='openie', cache_location=local + 'openie_cache_v1')
        # elif \
        assert self.entity_detection_model_type == OVERLAPPING_ENTITIES_KEY
        self.entity_detection_model = EntityDetectionOverlap(etype=entity_detection_model_type)

    def fnc(self, row):

        i, row = row
        opts = {}
        df_correct_choice = None
        option_correct_choice = None
        correct_choice = row[ 'answerKey' ]
        for c in row[ 'question' ][ 'choices' ]:
            option = c[ "text" ]
            ques = row['question']['stem'].lower().strip()
            option = option.lower().strip()

            if c['df'] is None:
                c['df'] = ques + ' ' + option
                print("** df_was_none **")

            df = c['df']
            if correct_choice == c['label']:
                df_correct_choice = df
                option_correct_choice = option
            for j,chain in enumerate(c['chains']):
                if self.entity_detection_model_type == "openie":
                    raise NotImplementedError
                elif self.entity_detection_model_type == OVERLAPPING_ENTITIES_KEY:
                    f1, f2 = chain[ 0 ][ 'text' ], chain[ 1 ][ 'text' ]
                    vals = {'fact1': f1, 'fact2': f2, 'cf': df, 'id': row[ 'id' ], 'option': option, 'opts': opts}
                    pattern = self.entity_detection_model.get_pattern(vals)
                    chain2 = {} #chain[2]
                    chain2[OVERLAPPING_ENTITIES_KEY] = {'txt12_candidates': pattern[ 'txt12_candidates' ],
                                                         'txt1c_candidates': pattern[ 'txt1c_candidates' ],
                                                         'txt2c_candidates': pattern[ 'txt2c_candidates' ]
                                                         }
                    c[ 'chains' ][ j ] = chain[ 0 ], chain[ 1 ], chain2
                else:
                    raise NotImplementedError
                print(" ---->>> i = ", i, " j=", j, "chain = ", json.dumps(c[ 'chains' ][ j ], indent=4))
                # print(json.dumps(c, indent=4))
                # break
            # break
        df = df_correct_choice
        f1 = row[ 'fact1' ]
        f2 = row[ 'fact2' ]
        option = option_correct_choice
        if self.entity_detection_model_type == OVERLAPPING_ENTITIES_KEY:
            vals = {'fact1': f1, 'fact2': f2, 'cf': df, 'id': row[ 'id' ], 'option': option, 'opts': opts}
            pattern = self.entity_detection_model.get_pattern(vals)
            row[OVERLAPPING_ENTITIES_KEY] = {'txt12_candidates': pattern[ 'txt12_candidates' ],
                                                       'txt1c_candidates': pattern[ 'txt1c_candidates' ],
                                                       'txt2c_candidates': pattern[ 'txt2c_candidates' ]
                                                       }
        return row

