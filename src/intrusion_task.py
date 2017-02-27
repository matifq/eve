import json
import operator, itertools
import numpy as np
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
    def prepare_queries(data_dict, description, similar_items=4):
        _query_lst = []
        for _type, item_lst in data_dict.items():
            possible_combinations = list(itertools.combinations(item_lst, similar_items))
            # print(_type)
            for selected_items in possible_combinations:
                for intruder_type, intruder_item_lst in data_dict.items():
                    if intruder_type == _type:
                        continue
                    for intruder_item in intruder_item_lst:
                        entity_list = list(selected_items)
                        entity_list.append(intruder_item)
                        # if 'Aloo tikki' in entity_list:
                        #     print(entity_list, intruder_item)
                        # print(entity_list)
                        _query = IntruderQuery(
                            entity_list,
                            intruder_item)
                        _query.set_description(description + '/' + _type)
                        _query_lst.append(_query)
                        # print(_query.get_description())
        return _query_lst


class IntruderQuery:
    def __init__(self, entity_list, correct_answer):
        self.entities_lst = list(set(entity_list))
        self.correct_answer = correct_answer
        self.description = None
        self.discovered_answer = None
        self.answered_correctly = None
        self.answer_dict = None

    def prepare_null_vectors(self, gram_to_vector):
        for entity in self.entities_lst:
            gram_to_vector[entity] = None
        return None

    def set_description(self, description):
        self.description = description

    def get_description(self):
        if self.description is None:
            return 'Entities:' + ', '.join(list(self.entities_lst))
        else:
            return self.description + ' - Detect the "intruder":'

    def get_all_correct_answer(self):
        return {self.correct_answer.lower()}

    def get_query(self):
        return self.entities_lst

    def set_discovered_answer(self, answer_dict, echo=True):
        self.answer_dict = answer_dict
        # print(answer_dict)
        sorted_answer_lst = sorted(answer_dict.items(), key=operator.itemgetter(1), reverse=False)
        answer = sorted_answer_lst[0][0].title()
        if echo:
            print(sorted_answer_lst)
            print([(entity, score/sorted_answer_lst[0][1]) for entity, score in sorted_answer_lst])
        self.discovered_answer = answer
        if answer.lower() in self.get_all_correct_answer():
            self.answered_correctly = True
        else:
            self.answered_correctly = False

    def get_discovered_answer(self):
        return self.discovered_answer

    def get_answered_status(self):
        return self.answered_correctly


class IntruderTask:
    def __init__(self):
        pass

    @staticmethod
    def explain_result(query, _index, entities_lst, max_n=-1):
        dimensions = _index.get_eve_dimensions()
        guessed_answer = query.get_discovered_answer()
        print('\n', query.get_description(), 'Known Answer:', query.get_all_correct_answer())
        print(' >', guessed_answer, query.get_answered_status())
        guessed_answer = guessed_answer.lower()
        cluster_matrix = []
        intruder_array = None
        for i, e in enumerate(entities_lst):
            print(e)
            v = _index.get_eve_vector(e)
            if e.lower() == guessed_answer:
                intruder_array = np.array(v)
                continue
            cluster_matrix.append(v)
        centroid_cluster = np.array(cluster_matrix).mean(axis=0)
        cluster_leftover = centroid_cluster - intruder_array
        intruder_leftover = intruder_array - centroid_cluster
        cluster_features = sorted(list(zip(*[cluster_leftover, dimensions])), reverse=True)
        intruder_features = sorted(list(zip(*[intruder_leftover, dimensions])), reverse=True)
        if max_n > -1:
            cluster_features = cluster_features[:max_n]
            intruder_features = intruder_features[:max_n]
        print('non-intruder features:', cluster_features)
        print('intruder features:', intruder_features)


    def run_tests_eve(self, _query_lst, _index, echo=True, explain_wrong=True, explain_always=False,
                      final_result_echo=False):
        results_lst = list()
        print('eve tests')
        results_lst.append(
            self.__run_test_eve__(_query_lst, _index,
                                  echo=echo, final_result_echo=final_result_echo,
                                  explain_wrong=explain_wrong,
                                  explain_always=explain_always))
        return results_lst

    @staticmethod
    def __run_test_eve__(_query_lst, _index,
                         echo=True, final_result_echo=False,
                         explain_wrong=True, explain_always=False, cache=True):
        cached_dict = dict()
        total_correct_answer = 0
        algo_results = list()
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_eve_vector(e1)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    v2 = _index.get_eve_vector(e2)
                    if cache and (e1, e2,) in cached_dict:
                        score = cached_dict[(e1, e2,)]
                    else:
                        score = functions.calculate_cosine_similarity(v1, v2)
                        cached_dict[(e1, e2,)] = score
                        cached_dict[(e2, e1,)] = score
                    # print(score)
                    scores_dict[e1] += score
                    scores_dict[e2] += score
            query.set_discovered_answer(scores_dict, echo=echo)
            if echo:
                print(' >', query.get_discovered_answer(), query.get_answered_status())
            if query.get_answered_status():
                total_correct_answer += 1
            if (explain_wrong and not query.get_answered_status()) or explain_always:
                IntruderTask.explain_result(query, _index, entities_lst, max_n=40)

            algo_results.append((query.get_query(), query.get_answered_status()))
        if echo or final_result_echo:
            print('Total Correct Answer', total_correct_answer, 'Out of', len(_query_lst),
                  total_correct_answer / len(_query_lst))
            print('')
        return total_correct_answer, len(_query_lst), algo_results

    def run_tests_word2vec(self, _query_lst, sg=None, echo=True, cache=True):
        similarity_cached_dict = dict()
        results_lst = list()
        print('word2vec tests, sg =', sg)
        total_correct_answer = 0
        algo_results = list()
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_word2vec_vector(e1, sg)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in similarity_cached_dict:
                        sim_score = similarity_cached_dict[(e1, e2,)]
                    else:
                        v2 = _index.get_word2vec_vector(e2, sg)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        similarity_cached_dict[(e1, e2,)] = sim_score
                        similarity_cached_dict[(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            if echo:
                print(' >', query.get_discovered_answer(), query.get_answered_status())
            if query.get_answered_status():
                total_correct_answer += 1

            algo_results.append((query.get_query(), query.get_answered_status()))
        if echo:
            print('Total Correct Answer', total_correct_answer, 'Out of', len(_query_lst),
                  total_correct_answer / len(_query_lst))
        results_lst.append((total_correct_answer, len(_query_lst), algo_results))
        return results_lst

    def run_tests_fasttext(self, _query_lst, sg=None, echo=True, cache=True):
        similarity_cached_dict = dict()
        results_lst = list()
        print('fasttext tests, sg =', sg)
        total_correct_answer = 0
        algo_results = list()
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_fasttext_vector(e1, sg)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in similarity_cached_dict:
                        sim_score = similarity_cached_dict[(e1, e2,)]
                    else:
                        v2 = _index.get_fasttext_vector(e2, sg)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        similarity_cached_dict[(e1, e2,)] = sim_score
                        similarity_cached_dict[(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            if echo:
                print(' >', query.get_discovered_answer(), query.get_answered_status())
            if query.get_answered_status():
                total_correct_answer += 1

            algo_results.append((query.get_query(), query.get_answered_status()))
        if echo:
            print('Total Correct Answer', total_correct_answer, 'Out of', len(_query_lst),
                  total_correct_answer / len(_query_lst))
        results_lst.append((total_correct_answer, len(_query_lst), algo_results))
        return results_lst

    def run_tests_glove(self, _query_lst, echo=True, cache=True):
        similarity_cached_dict = dict()
        results_lst = list()
        print('glove tests')
        total_correct_answer = 0
        algo_results = list()
        for question_no, (query) in enumerate(_query_lst):
            if echo:
                print('Question:' + str(question_no + 1) + '/' + str(len(_query_lst)),
                      query.get_description(), 'Known Answer:', query.get_all_correct_answer())
            scores_dict = dict()
            entities_lst = query.get_query()
            for entity in entities_lst:
                scores_dict[entity] = 0.0
            for i, e1 in enumerate(entities_lst):
                v1 = _index.get_glove_vector(e1)
                for j in range(i + 1, len(entities_lst)):
                    e2 = entities_lst[j]
                    if cache and (e1, e2,) in similarity_cached_dict:
                        sim_score = similarity_cached_dict[(e1, e2,)]
                    else:
                        v2 = _index.get_glove_vector(e2)
                        sim_score = functions.calculate_cosine_similarity(v1, v2)
                        similarity_cached_dict[(e1, e2,)] = sim_score
                        similarity_cached_dict[(e2, e1,)] = sim_score
                    scores_dict[e1] += sim_score
                    scores_dict[e2] += sim_score
            query.set_discovered_answer(scores_dict, echo=echo)
            if echo:
                print(' >', query.get_discovered_answer(), query.get_answered_status())
            if query.get_answered_status():
                total_correct_answer += 1
            algo_results.append((query.get_query(), query.get_answered_status()))
        if echo:
            print('Total Correct Answer', total_correct_answer, 'Out of', len(_query_lst),
                  total_correct_answer / len(_query_lst))
        results_lst.append((total_correct_answer, len(_query_lst), algo_results))
        return results_lst


def test_explain():
    _index = Index(config)

    q1 = IntruderQuery(['Hawk', 'Penguin', 'Gull', 'Parrot', 'Snake'], 'Snake')
    q1.set_description('Animal Class/Birds')

    q2 = IntruderQuery(['I Am Legend (film)', 'Insidious (film)', 'A Nightmare on Elm Street',
                        'Final Destination (film)', 'Children of Men'],'Children of Men')
    q2.set_description('Movie Genre/Horror Film')

    task = IntruderTask()

    _index.load_all_indexes('animal_classes')
    result_lst = task.run_tests_eve([q1], _index, echo=False, explain_wrong=True, explain_always=True)
    for i, (result) in enumerate(result_lst):
        correct, total, eve_results = result
        print('-> Total Correct Answer', correct, 'Out of', total, correct / total)

    _index.load_all_indexes('movie_genres')
    result_lst = task.run_tests_eve([q2], _index, echo=False, explain_wrong=True, explain_always=True)
    for i, (result) in enumerate(result_lst):
        correct, total, eve_results = result
        print('-> Total Correct Answer', correct, 'Out of', total, correct / total)


if __name__ == '__main__':
    types_lst = ['cuisine', 'nobel_laureates', 'music_genres', 'movie_genres', 'european_cities', 'animal_classes',
                 'country_continent']
    # types_lst = ['cuisine']
    first_n_items, similar_items = 50, 4
    query_lst = list()
    _type_query_lst = list()
    for __type in types_lst:
        data_file = ReadDataFile(__type, first_n_items=first_n_items)
        automatic_queries = \
            AutomateQuery().prepare_queries(data_file.get_dict(), data_file.get_description(), similar_items=similar_items)
        query_lst.extend(automatic_queries)
        _type_query_lst.append(automatic_queries)
    print('total queries', len(query_lst))
    _index = Index(config)
    task = IntruderTask()
    for i, query_lst in enumerate(_type_query_lst):
        _index.load_all_indexes(types_lst[i])
        results_dict = dict()
        result_lst = task.run_tests_eve(query_lst, _index, echo=False, final_result_echo=False, explain_wrong=False, explain_always=False)
        for result in result_lst:
            correct, total, eve_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['eve'] = list(zip(*eve_results))[1]
            results_dict['eve'] = eve_results
            results_dict['_queries_'] = list(zip(*eve_results))[0]
        result_lst = task.run_tests_word2vec(query_lst, sg=0, echo=False)
        for result in result_lst:
            correct, total, word2vec_cbow_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['word2vec_cbow'] = list(zip(*word2vec_cbow_results))[1]
            results_dict['word2vec_cbow'] = word2vec_cbow_results
        result_lst = task.run_tests_word2vec(query_lst, sg=1, echo=False)
        for result in result_lst:
            correct, total, word2vec_sg_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['word2vec_sg'] = list(zip(*word2vec_sg_results))[1]
            results_dict['word2vec_sg'] = word2vec_sg_results
        result_lst = task.run_tests_fasttext(query_lst, sg=0, echo=False)
        for result in result_lst:
            correct, total, fasttext_cbow_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['fasttext_cbow'] = list(zip(*fasttext_cbow_results))[1]
            results_dict['fasttext_cbow'] = fasttext_cbow_results
        result_lst = task.run_tests_fasttext(query_lst, sg=1, echo=False)
        for result in result_lst:
            correct, total, fasttext_sg_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['fasttext_sg'] = list(zip(*fasttext_sg_results))[1]
            results_dict['fasttext_sg'] = fasttext_sg_results
        result_lst = task.run_tests_glove(query_lst, echo=False)
        for result in result_lst:
            correct, total, glove_results = result
            print(types_lst[i].title(), '-> Total Correct Answer', correct, 'Out of', total, correct / total)
            results_dict['glove'] = glove_results
        json.dump(results_dict, open(config.base_path + 'output/intrusion/results-for-tests-' + types_lst[i] + '.json', 'w'))
    print('> explanation part')
    test_explain()