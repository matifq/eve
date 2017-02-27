import operator
import numpy as np
from shared import config
from shared.indexes import Index
from shared import functions


class ReadDataFile:

    def __init__(self, _type, first_n_items=20, echo=True):
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
            if echo:
                print(_key, len(self._dict[_key]))

    def get_dict(self):
        return self._dict

    def get_description(self):
        return self._description


class AutomateQuery:

    def __init__(self):
        pass

    @staticmethod
    def prepare_queries(data_dict, description):
        _query_lst = list()
        item_list = list()
        for _category_label, selected_items in data_dict.items():
            # print(selected_items)
            # print('->', _category_label)
            correct_answer_set = selected_items
            item_list.extend(list(selected_items))
            _query = CategoryQuery(_category_label, correct_answer_set)
            _query.set_description(description + '/' + _category_label)
            _query_lst.append(_query)
        for _query in _query_lst:
            _query.add_item_list(item_list)
        return _query_lst


class CategoryQuery:
    def __init__(self, category_name, correct_answer_set):
        self.query = category_name
        self.correct_answer_set = {_ans.lower() for _ans in correct_answer_set}
        self.item_list = None
        self.discovered_answer = None
        self.description = None
        self.answer_dict = None
        self.measure_p_at_k = None
        self.measure_ap = None
        self.incorrect_answers = None

    def add_item_list(self, item_list):
        self.item_list = list(set(item_list))

    def get_possible_answer_set(self):
        return self.item_list

    def prepare_null_vectors(self, gram_to_vector):
        gram_to_vector[self.query] = None
        for item in self.item_list:
            gram_to_vector[item] = None
        return None

    def set_description(self, description):
        self.description = description

    def get_description(self):
        if self.description is None:
            return 'Sort item list for Category Name:' + self.query
        else:
            return self.description + ' - Sort relevant item list:'

    def get_all_correct_answer(self):
        return self.correct_answer_set

    def get_query(self):
        return self.query

    def set_discovered_answer(self, answer_dict, echo=True):
        self.answer_dict = answer_dict
        sorted_answer_lst = sorted(answer_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.incorrect_answers = []
        # print(sorted_answer_lst)
        k = len(self.get_all_correct_answer())
        # print('total correct answers', k)
        self.measure_p_at_k = 0
        self.measure_ap = 0
        num_correct_answers = 0
        last_correct_index = 0
        for i, (query_answer_tuple, score) in enumerate(sorted_answer_lst):
            query, answer = query_answer_tuple
            # print(i, k, i < k)
            # print(answer.lower(), self.get_all_correct_answer())
            if answer.lower() in self.get_all_correct_answer():
                num_correct_answers += 1
                if i < k:
                    self.measure_p_at_k += 1
                self.measure_ap += num_correct_answers / (i + 1)
                last_correct_index = i + 1
            elif k > num_correct_answers:
                self.incorrect_answers.append((answer, i + 1))
        self.measure_p_at_k /= k  # averaged over k
        self.measure_ap /= k  # divided by number of correct results
        answers, scores = list(zip(*sorted_answer_lst[:last_correct_index]))
        self.discovered_answer = list(zip(*[list(zip(*answers))[1], scores]))
        if echo:
            print('p@k', self.measure_p_at_k, 'ap', self.measure_ap, sorted_answer_lst)

    def get_measures(self):
        return {'p@k': self.measure_p_at_k, 'ap': self.measure_ap}

    def get_answered_status(self):
        return 1.0 == self.measure_p_at_k

    def get_discovered_answer(self):
        return self.discovered_answer

    def get_incorrect_answers(self):
        return self.incorrect_answers


class RankingTask:
    def __init__(self):
        pass

    @staticmethod
    def explain_result(v1, e1, query, _index, dimensions, max_n=-1, top_n_answers=3):
        incorrect_answers = query.get_incorrect_answers()
        answers = query.get_discovered_answer()
        print('\n', query.get_description(), 'Known Answer:', [ans.title() for ans in query.get_all_correct_answer()])
        print(' > sorted order:', answers)
        print(' > incorrect answers:', incorrect_answers)
        meta_vector = np.array(v1) * 1.0
        weight = 1.0
        for e, score in answers:
            v = _index.get_eve_vector(e)
            meta_vector += np.array(v) * score
            weight += score
        meta_vector /= weight  # weighted average
        for e2, rank in incorrect_answers[:top_n_answers]:
            e2 = e2
            v2 = _index.get_eve_vector(e2)
            vector_product = np.array(meta_vector) * np.array(v2)
            dominant_features = list()
            for score, dim in sorted(list(zip(*[vector_product, dimensions])), reverse=True):
                if score == 0:
                    break
                dominant_features.append((score, dim))
            print(' --> dominant features (' + e1 + ', ' + e2 + ')', dominant_features[:max_n])
        print('> top answers')
        for e, score in answers[:top_n_answers]:
            e2 = e
            v2 = _index.get_eve_vector(e2)
            vector_product = np.array(meta_vector) * np.array(v2)
            dominant_features = list()
            for score, dim in sorted(list(zip(*[vector_product, dimensions])), reverse=True):
                if score == 0:
                    break
                dominant_features.append((score, dim))
            print(' --> dominant features (' + e1 + ', ' + e2 + ')', dominant_features[:max_n])

    def run_tests_eve(self, _query_lst, _index, echo=True, explain_wrong=True, explain_always=False,
                      max_n_features=40, top_n_answers=3):
        results_lst = []
        print('eve tests')
        results_lst.append(
            self.__run_test_eve__(_query_lst, _index, echo=echo, explain_wrong=explain_wrong,
                                  explain_always=explain_always, max_n_features=max_n_features,
                                  top_n_answers=top_n_answers))
        return results_lst

    @staticmethod
    def __run_test_eve__(_query_lst, _index, echo=True, explain_wrong=True, explain_always=False, cache=True,
                         max_n_features=40, top_n_answers=3):
        similarity_cached_dict = dict()
        _measures = {'p@k': 0, 'ap': 0}
        dimensions = _index.get_eve_dimensions()
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            e1 = query.get_query()
            v1 = _index.get_eve_vector(e1)
            for e2 in query.get_possible_answer_set():
                v2 = _index.get_eve_vector(e2)
                if cache and (e1, e2,) in similarity_cached_dict:
                    score = similarity_cached_dict[(e1, e2,)]
                else:
                    score = functions.calculate_cosine_similarity(v1, v2)
                    similarity_cached_dict[(e1, e2,)] = score
                    similarity_cached_dict[(e2, e1,)] = score
                scores_dict[e1, e2] = score
            query.set_discovered_answer(scores_dict, echo=echo)
            for k, v in query.get_measures().items():
                _measures[k] += v
            if echo:
                print(' >', query.get_measures())
            if (explain_wrong and not query.get_answered_status()) or explain_always:
                RankingTask.explain_result(v1, e1, query, _index, dimensions,
                                           max_n=max_n_features, top_n_answers=top_n_answers)
        if echo:
            print('Measure: ' + ','.join([k + ': ' + str(v/len(_query_lst)) for k, v in _measures.items()]),
                  'Total Queries:', len(_query_lst))
            print('')
        return _measures, len(_query_lst)

    def run_tests_word2vec(self, _query_lst, sg=None, echo=True, cache=True):
        similarity_cached_dict = dict()
        _measures = {'p@k': 0, 'ap': 0}
        results_lst = list()
        print('word2vec tests, sg =', sg)
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            e1 = query.get_query()
            v1 = _index.get_word2vec_vector(e1, sg)
            for e2 in query.get_possible_answer_set():
                if cache and (e1, e2,) in similarity_cached_dict:
                    sim_score = similarity_cached_dict[(e1, e2,)]
                else:
                    v2 = _index.get_word2vec_vector(e2, sg)
                    sim_score = functions.calculate_cosine_similarity(v1, v2)
                    similarity_cached_dict[(e1, e2,)] = sim_score
                    similarity_cached_dict[(e2, e1,)] = sim_score
                scores_dict[e1, e2] = sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            for k, v in query.get_measures().items():
                _measures[k] += v
            if echo:
                print(' >', query.get_measures())
        if echo:
            print('Measure: ' + ','.join([k + ': ' + str(v / len(_query_lst)) for k, v in _measures.items()]),
                  'Total Queries:', len(_query_lst))
            print('')
        results_lst.append((_measures, len(_query_lst)))
        return results_lst

    def run_tests_fasttext(self, _query_lst, sg=None, echo=True, cache=True):
        similarity_cached_dict = dict()
        _measures = {'p@k': 0, 'ap': 0}
        results_lst = list()
        print('fasttext tests, sg =', sg)
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            e1 = query.get_query()
            v1 = _index.get_fasttext_vector(e1, sg)
            for e2 in query.get_possible_answer_set():
                if cache and (e1, e2,) in similarity_cached_dict:
                    sim_score = similarity_cached_dict[(e1, e2,)]
                else:
                    v2 = _index.get_fasttext_vector(e2, sg)
                    sim_score = functions.calculate_cosine_similarity(v1, v2)
                    similarity_cached_dict[(e1, e2,)] = sim_score
                    similarity_cached_dict[(e2, e1,)] = sim_score
                scores_dict[e1, e2] = sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            for k, v in query.get_measures().items():
                _measures[k] += v
            if echo:
                print(' >', query.get_measures())
        if echo:
            print('Measure: ' + ','.join([k + ': ' + str(v / len(_query_lst)) for k, v in _measures.items()]),
                  'Total Queries:', len(_query_lst))
            print('')
        results_lst.append((_measures, len(_query_lst)))
        return results_lst

    def run_tests_glove(self, _query_lst, echo=True, cache=True):
        similarity_cached_dict = dict()
        _measures = {'p@k': 0, 'ap': 0}
        results_lst = list()
        print('glove tests')
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            e1 = query.get_query()
            v1 = _index.get_glove_vector(e1)
            for e2 in query.get_possible_answer_set():
                if cache and (e1, e2,) in similarity_cached_dict:
                    sim_score = similarity_cached_dict[(e1, e2,)]
                else:
                    v2 = _index.get_glove_vector(e2)
                    sim_score = functions.calculate_cosine_similarity(v1, v2)
                    similarity_cached_dict[(e1, e2,)] = sim_score
                    similarity_cached_dict[(e2, e1,)] = sim_score
                scores_dict[e1, e2] = sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            for k, v in query.get_measures().items():
                _measures[k] += v
            if echo:
                print(' >', query.get_measures())
        if echo:
            print('Measure: ' + ','.join([k + ': ' + str(v / len(_query_lst)) for k, v in _measures.items()]),
                  'Total Queries:', len(_query_lst))
            print('')
        results_lst.append((_measures, len(_query_lst)))
        return results_lst


def test_explain():
    _index = Index(config)
    _types_lst = ['nobel_laureates', 'music_genres']
    _query_lst = list()
    _type_query_lst = list()
    allowed_queries = {'Nobel laureates/List of Nobel laureates in Chemistry - Sort relevant item list:',
                       'Music genres/Classical music - Sort relevant item list:'}
    for __type in _types_lst:
        data_file = ReadDataFile(__type, first_n_items=20, echo=False)
        automatic_queries = \
            AutomateQuery().prepare_queries(data_file.get_dict(), data_file.get_description())
        _query_lst.extend(automatic_queries)
        _type_query_lst.append(automatic_queries)

    _task = RankingTask()
    for _i, _query_lst in enumerate(_type_query_lst):
        _index.load_all_indexes(_types_lst[_i])
        _allowed_query_lst = list()
        for _q in _query_lst:
            if _q.get_description() in allowed_queries:
                _allowed_query_lst.append(_q)
        result_lst = _task.run_tests_eve(_allowed_query_lst, _index, echo=False, explain_wrong=True,
                                         explain_always=True, max_n_features=6, top_n_answers=1)
        for result in result_lst:
            _measures, _total = result
            print(_types_lst[_i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / _total) for k, v in _measures.items()]),
                  'Total Queries:', _total)


if __name__ == '__main__':
    types_lst = ['cuisine', 'nobel_laureates', 'music_genres', 'movie_genres', 'european_cities', 'animal_classes',
                 'country_continent']
    # types_lst = ['cuisine']
    query_lst = list()
    _type_query_lst = list()
    for __type in types_lst:
        data_file = ReadDataFile(__type, first_n_items=50)
        automatic_queries = \
            AutomateQuery().prepare_queries(data_file.get_dict(), data_file.get_description())
        query_lst.extend(automatic_queries)
        _type_query_lst.append(automatic_queries)

    print('total queries', len(query_lst))
    _index = Index(config)
    task = RankingTask()
    for i, query_lst in enumerate(_type_query_lst):
        _index.load_all_indexes(types_lst[i])
        result_lst = task.run_tests_eve(query_lst, _index, echo=False, explain_wrong=False, explain_always=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
        result_lst = task.run_tests_word2vec(query_lst, sg=0, echo=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
        result_lst = task.run_tests_word2vec(query_lst, sg=1, echo=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
        result_lst = task.run_tests_fasttext(query_lst, sg=0, echo=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
        result_lst = task.run_tests_fasttext(query_lst, sg=1, echo=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
        result_lst = task.run_tests_glove(query_lst, echo=False)
        for result in result_lst:
            measures, total = result
            print(types_lst[i].title(),
                  ' -> Measure: ' + ', '.join([k + ': ' + str(v / total) for k, v in measures.items()]),
                  'Total Queries:', total)
    print('> explanation part')
    test_explain()