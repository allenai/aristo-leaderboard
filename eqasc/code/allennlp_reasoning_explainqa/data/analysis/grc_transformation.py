# coding: utf-8
import string
from allennlp_reasoning_explainqa.common.constants import *

def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.strip()


class DisjointSet(object):

    def __init__(self):
        self.leader = {}  # maps a member to the group's leader
        self.group = {}  # maps a group leader to the group (which is a set)

    def add_ele(self, a):
        assert a not in self.group
        self.leader[a] = a
        self.group[a] = set([a])

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])



class GRCTransform:

    def __init__(self,
                 entity_detection_model_type: str = OVERLAPPING_ENTITIES_KEY,
                 use_tag_representation=True,
                 use_grouping=True,
                 use_tag_representation_type='abstract'):

        self.entity_detection_model_type = entity_detection_model_type
        self.use_tag_representation = use_tag_representation
        self._use_grouping = use_grouping
        self.use_tag_representation_type = use_tag_representation_type


    def _apply_mapper_to_string(self, s, e, entity_mapper):
        # print("entity_mapper = " ,entity_mapper )
        locations = []
        idx = 0
        processed_till_now = 0
        sret = ""
        while idx < len(s):
            idx = s.find(e, idx)
            if idx == -1:
                break
            # prefix  = s[:idx] + e + '_start'
            jdx = idx + len(e)
            if s.find(' ', jdx) >= 0:
                jdx = s.find(' ', jdx)
            else:
                jdx = len(s)
            locations.append([idx, jdx])
            if self.use_tag_representation_type == "default":
                sret = sret + s[processed_till_now:idx] + ' ' + entity_mapper[e.lower()] + ' ' + s[idx:jdx] + ' ' + \
                       entity_mapper[e.lower()] + ' '
            elif self.use_tag_representation_type == "abstract":
                sret = sret + s[processed_till_now:idx] + ' ' + entity_mapper[e.lower()] + ' '
            else:
                assert False, "Not supported option for use_tag_representation_type"
            idx = jdx + 1
            processed_till_now = jdx
        if processed_till_now != len(s):
            sret = sret + s[processed_till_now:]
        return sret

    def _group_entities(self, lst_of_entities):
        # par = [i for i,e in enumerate(lst_of_entities) ]
        ds = DisjointSet()
        for i, val in enumerate(lst_of_entities):
            ds.add_ele(i)
        for i, val in enumerate(lst_of_entities):
            for j, val2 in enumerate(lst_of_entities):
                if val != val2 and val2.count(val) > 0:
                    ds.add(i, j)
        ret = []
        for leader, group_vals in ds.group.items():
            tmp = [lst_of_entities[j] for j in group_vals]
            tmp = sorted(tmp, key=lambda x: -len(x))
            ret.append(tmp)
        return ret

    def get_grc_form(self, chain, debug=False):
        pattern = chain[2][OVERLAPPING_ENTITIES_KEY]
        lst_of_sentences = [ normalize_text(chain[0]['text']),
                             normalize_text(chain[1]['text']),
                             normalize_text(chain[2]['df']) ]
        import copy
        if self.entity_detection_model_type == 'openie':
            assert False

        elif self.entity_detection_model_type == OVERLAPPING_ENTITIES_KEY:
            lst_of_entities = []
            ret = [copy.deepcopy(s) for s in lst_of_sentences]
            for k in pattern:
                for kk in pattern[k].values():
                    lst_of_entities.extend(kk)
            if debug:
                print("=====>>> lst_of_entities = ", lst_of_entities)
            entity_mapper = {}
            if self._use_grouping:
                lst_of_entities_grouped = self._group_entities(lst_of_entities)
                lst_of_entities = []
                for i, e in enumerate(lst_of_entities_grouped):
                    for ee in e:
                        entity_mapper[ee.lower()] = '[unused' + str(i) + ']'
                    lst_of_entities.extend(e)
            else:
                entity_mapper = {e.lower(): '[unused' + str(i) + ']' for i, e in enumerate(lst_of_entities)}
            if debug:
                print("=====>>> entity_mapper = ", entity_mapper)
            predicate_seq = []
            for j, s in enumerate(ret):
                for e in lst_of_entities:
                    if self.use_tag_representation is not None:
                        sret = self._apply_mapper_to_string(s, e, entity_mapper)
                        s = sret
                predicate_seq.append(s)
            # predicate_seq = self._get_predicate(lst_of_entities, lst_of_sentences=lst_of_sentences)
            assert len(predicate_seq) == len(lst_of_sentences)
            return predicate_seq
