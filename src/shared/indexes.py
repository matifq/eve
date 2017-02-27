import json


class Index:

    def __init__(self, config):
        self.eve_index_path = config.eve_index_path
        self.word2vec_index_path = config.word2vec_index_path
        self.fasttext_index_path = config.fasttext_index_path
        self.glove_index_path = config.glove_index_path
        self.eve_cached_dict = None
        self.eve_cached_dict = None

    def load_eve(self, _type):
        self.eve_cached_dict = json.load(open(self.eve_index_path + 'eve_' + _type + '.json', 'r'))

    def load_word2vec(self, _type):
        self.word2vec_cbow_cached_dict = json.load(
            open(self.word2vec_index_path + 'word2vec_cbow_' + _type + '.json', 'r'))
        self.word2vec_sg_cached_dict = json.load(open(self.word2vec_index_path + 'word2vec_sg_' + _type + '.json', 'r'))

    def load_fasttext(self, _type):
        self.fasttext_cbow_cached_dict = json.load(open(self.fasttext_index_path + 'fasttext_cbow_' + _type + '.json', 'r'))
        self.fasttext_sg_cached_dict = json.load(open(self.fasttext_index_path + 'fasttext_sg_' + _type + '.json', 'r'))

    def load_glove(self, _type):
        self.glove_cached_dict = json.load(open(self.glove_index_path + 'glove_' + _type + '.json', 'r'))

    def get_eve_vector(self, e):
        return self.eve_cached_dict[e]

    def get_word2vec_vector(self, e, sg):
        if sg == 0:
            return self.word2vec_cbow_cached_dict[e]
        else:
            return self.word2vec_sg_cached_dict[e]

    def get_fasttext_vector(self, e, sg):
        if sg == 0:
            return self.fasttext_cbow_cached_dict[e]
        else:
            return self.fasttext_sg_cached_dict[e]


    def get_glove_vector(self, e):
        return self.glove_cached_dict[e]

    def get_eve_dimensions(self):
        return self.eve_cached_dict['__dim__']

    def load_all_indexes(self, _type):
        self.load_eve(_type)
        self.load_word2vec(_type)
        self.load_fasttext(_type)
        self.load_glove(_type)


