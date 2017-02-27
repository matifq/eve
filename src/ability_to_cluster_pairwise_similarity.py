import itertools
import numpy as np
import json
from shared import config
from shared.indexes import Index
from shared import functions


class ReadDataFile:

    def __init__(self, _type, first_n_items=20):
        self._dict = dict()
        f = open(config.task_dataset_path + _type + '.txt', 'r')
        lines = f.read().split('\n')
        self._description = lines[0].strip()
        for line in lines[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            _key, value_lst = (line.split(':', 1))
            _key = _key.strip()
            self._dict[_key] = list()
            for value in value_lst.strip().split(';')[:first_n_items]:
                if len(value):
                    self._dict[_key].append(value)
            self._dict[_key] = set(self._dict[_key])
            print(_key, len(self._dict[_key]))

    def get_dict(self):
        return self._dict

    def get_description(self):
        return self._description


class AutomateQuery:

    def __init__(self):
        pass

    @staticmethod
    def prepare_queries(data_dict, description, first_n_items=1):
        _query_lst = []
        for _type, item_lst in data_dict.items():
            possible_combinations = list(itertools.combinations(item_lst, first_n_items))
            # print(_type)
            # print(item_lst, possible_combinations)
            for selected_items in possible_combinations:
                for iter_type, iter_item_lst in data_dict.items():
                    for intruder_item in iter_item_lst:
                        entity_list = list(selected_items)
                        entity_list.append(intruder_item)
                        _query = ClusterQuery(
                            entity_list,
                            intruder_item)
                        _query.set_description(description + '/' + _type)
                        _query_lst.append(_query)
                        # print(_query.get_description())
        return _query_lst


class ClusterQuery:
    def __init__(self, entity_list, correct_answer):
        self.entities_lst = list(set(entity_list))
        self.correct_answer = correct_answer
        self.description = None
        self.discovered_answer = None
        self.answered_correctly = None
        self.answer_dict = None

    def set_description(self, description):
        self.description = description

    def get_description(self):
        if self.description is None:
            return 'Entities:' + ', '.join(list(self.entities_lst))
        else:
            return self.description + ' - Ability to Cluster":'

    def get_query(self):
        return self.entities_lst


class ClusterTask:
    def __init__(self, cluster_dict):
        self.cached_eve_dict = None
        self.word2vec_cached_dict = None
        self.fasttext_cached_dict = None
        self.glove_cached_dict = None
        self.cluster_dict = cluster_dict
        self.init_cache()
        pass

    def init_cache(self):
        self.cached_eve_dict = dict()
        self.word2vec_cached_dict = [dict(), dict()]
        self.fasttext_cached_dict = [dict(), dict()]
        self.glove_cached_dict = dict()

    def explain_cluster_eve(self, _type, _index, dimensions):
        print('explain', _type)
        centroid_category_cluster = {}
        type_matrix = []
        for category, item_list in self.cluster_dict[_type].items():
            # print(category, item_list)
            # for item in item_list:
            #     print(item, item in gram_to_vector)
            matrix_category = [_index.get_eve_vector(item) for item in item_list]
            type_matrix.extend(matrix_category)
            centroid_category_cluster[category] = np.array(matrix_category).mean(axis=0)
        centroid_main_cluster = np.array(type_matrix).mean(axis=0)
        main_type_features = sorted(list(zip(*[centroid_main_cluster, dimensions])), reverse=True)
        print('Overall space:', _type, main_type_features[:10])
        for category, category_centroid in centroid_category_cluster.items():
            left_over_array = category_centroid - centroid_main_cluster
            leftover_features = sorted(list(zip(*[left_over_array, dimensions])), reverse=True)
            print('Category:', category, leftover_features[:10])

    def run_tests_eve(self, _type, _query_lst, _index):
        results_lst = []
        print('eve tests')
        results_lst.append(
            self.__run_test_eve__(_type, _query_lst, _index))
        return results_lst

    def __run_test_eve__(self, _type, _query_lst, _index, cache=True):
        dimensions = _index.get_eve_dimensions()
        for question_no, (query) in enumerate(_query_lst):
            entities_lst = query.get_query()
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_eve_vector(e1)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    v2 = _index.get_eve_vector(e2)
                    if cache and (e1, e2,) in self.cached_eve_dict:
                        score = self.cached_eve_dict[(e1, e2,)]
                    else:
                        score = functions.calculate_cosine_similarity(v1, v2)
                        self.cached_eve_dict[(e1, e2,)] = score
                        self.cached_eve_dict[(e2, e1,)] = score
                    # print(score)
        self.explain_cluster_eve(_type, _index, dimensions)
        return

    def get_similarity_matrix(self):
        word2vec_similarity_matrix = [None, None]
        fasttext_similarity_matrix = [None, None]
        eve_similarity_matrix = ({pair: value for pair, value in self.cached_eve_dict.items()})
        word2vec_similarity_matrix[0] = {pair: value for pair, value in self.word2vec_cached_dict[0].items()}
        word2vec_similarity_matrix[1] = {pair: value for pair, value in self.word2vec_cached_dict[1].items()}
        fasttext_similarity_matrix[0] = {pair: value for pair, value in self.fasttext_cached_dict[0].items()}
        fasttext_similarity_matrix[1] = {pair: value for pair, value in self.fasttext_cached_dict[1].items()}
        glove_similarity_matrix = {pair: value for pair, value in self.glove_cached_dict.items()}
        return eve_similarity_matrix, word2vec_similarity_matrix, fasttext_similarity_matrix, glove_similarity_matrix

    def run_tests_word2vec(self, _query_lst, sg=None, cache=True):
        print('word2vec tests, sg =', sg)
        for question_no, (query) in enumerate(_query_lst):
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_word2vec_vector(e1, sg)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in self.word2vec_cached_dict[sg]:
                        sim_score = self.word2vec_cached_dict[sg][(e1, e2,)]
                    else:
                        v2 = _index.get_word2vec_vector(e2, sg)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        self.word2vec_cached_dict[sg][(e1, e2,)] = sim_score
                        self.word2vec_cached_dict[sg][(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
        return

    def run_tests_fasttext(self, _query_lst, sg=None, cache=True):
        print('fasttext tests, sg =', sg)
        for question_no, (query) in enumerate(_query_lst):
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_fasttext_vector(e1, sg)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in self.fasttext_cached_dict[sg]:
                        sim_score = self.fasttext_cached_dict[sg][(e1, e2,)]
                    else:
                        v2 = _index.get_fasttext_vector(e2, sg)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        self.fasttext_cached_dict[sg][(e1, e2,)] = sim_score
                        self.fasttext_cached_dict[sg][(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
        return

    def run_tests_glove(self, _query_lst, cache=True):
        print('glove tests')
        for question_no, (query) in enumerate(_query_lst):
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_glove_vector(e1)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in self.glove_cached_dict:
                        sim_score = self.glove_cached_dict[(e1, e2,)]
                    else:
                        v2 = _index.get_glove_vector(e2)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        self.glove_cached_dict[(e1, e2,)] = sim_score
                        self.glove_cached_dict[(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
        return

if __name__ == '__main__':
    types_lst = ['cuisine', 'nobel_laureates', 'music_genres', 'movie_genres', 'european_cities', 'animal_classes',
                 'country_continent']
    # types_lst = ['cuisine']
    query_lst = list()
    _type_query_lst = list()
    cluster_dict = dict()
    for __type in types_lst:
        data_file = ReadDataFile(__type, first_n_items=50)
        automatic_queries = \
            AutomateQuery().prepare_queries(data_file.get_dict(), data_file.get_description(), first_n_items=1)
        query_lst.extend(automatic_queries)
        _type_query_lst.append((automatic_queries, __type))
        cluster_dict[__type] = data_file.get_dict()

    print('total queries', len(query_lst))
    _index = Index(config)
    task = ClusterTask(cluster_dict)
    for i, (query_lst, __type) in enumerate(_type_query_lst):
        _index.load_all_indexes(types_lst[i])
        task.init_cache()
        task.run_tests_eve(__type, query_lst, _index)
        task.run_tests_word2vec(query_lst, sg=0)
        task.run_tests_word2vec(query_lst, sg=1)
        task.run_tests_fasttext(query_lst, sg=0)
        task.run_tests_fasttext(query_lst, sg=1)
        task.run_tests_glove(query_lst)
        eve_similarity_matrix, word2vec_similarity_matrix, fasttext_similarity_matrix, glove_similarity_matrix = \
            task.get_similarity_matrix()
        json.dump({str(k): v for k, v in eve_similarity_matrix.items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_eve.json', 'w'))

        json.dump({str(k): v for k, v in word2vec_similarity_matrix[0].items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_word2vec_cbow.json', 'w'))

        json.dump({str(k): v for k, v in word2vec_similarity_matrix[1].items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_word2vec_sg.json', 'w'))

        json.dump({str(k): v for k, v in fasttext_similarity_matrix[0].items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_fasttext_cbow.json', 'w'))

        json.dump({str(k): v for k, v in fasttext_similarity_matrix[1].items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_fasttext_sg.json', 'w'))

        json.dump({str(k): v for k, v in glove_similarity_matrix.items()},
                  open(config.base_path + 'output/pairwise_similarity/' + types_lst[i] + '_glove.json', 'w'))
