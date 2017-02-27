import os, shutil
from shared import config


def create_folder(path_to_folder, msg=None):
    if os.path.isdir(path_to_folder):
        shutil.rmtree(path_to_folder)
    elif os.path.exists(path_to_folder[:-1]):
        os.remove(path_to_folder[:-1])
    os.mkdir(path_to_folder)
    if msg:
        print(msg)


def download_files():
    url_dataverse = 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/V7MDP6'
    print('Download data files from', url_dataverse)


def setup():
    create_folder(config.__dataset_path__, '* dataset folder initialised')
    create_folder(config.__output_path__, ' * output folder initialised')
    create_folder(config.__output_path__ + 'cluster-images/', ' * output/cluster-images folder initialised')
    create_folder(config.__output_path__ + 'intrusion/', ' * output/intrusion folder initialised')
    create_folder(config.__output_path__ + 'pairwise_similarity/', ' * output/pairwise_similarity folder initialised')
    download_files()

if __name__ == '__main__':
    setup()