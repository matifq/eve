# EVE: Explainable Vector Based Embedding Technique Using Wikipedia
## Authors of Research contributions: Muhammad Atif Qureshi, Derek Greene
### Tested: Python 3.4.3, Anaconda3 Jupyter Notebook 4.2.0
======
**Steps**
1  Download dataset files from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/V7MDP6 and extract them into the folder 'dataset'
  * After downloading and extraction, the structure shall look like  dataset/tasks/ and dataset/indexes/
2  From src, run intrusion_task.py --> Experiment 1: intrusion detection
  * It will reproduce results as shown in Table 4 and explanations as shown in Table 5 and 6 of the paper
3  Run Notebook from root folder, run ipython/intrusion_p-value.ipynb
  * It will reproduce p-values as discussed in Table 4 of the paper
4  From src, run ability_to_cluster_pairwise_similarity.py --> Experiment 2: ability to cluster
  * It will calculate pairwise scores which is used in later script and it will reproduce explanations as shown in Table 10 and 11 of the paper
5  Run Notebook from root folder, run ipython/cluster-validation-measures.ipynb
  * It will reproduce results as shown Table 7, 8, and 9 of the paper
6. Run Notebook from root folder, run ipython/cluster-visualisations.ipynb
  * It will reproduce visualisation as shown in Figure 4 of the paper.
7  From src, run sorting_relevant_items_first_task.py --> Experiment 3: sorting relevant items first
  * It will reproduce results as shown in Tables 12 and 13, and explanations as shown in Table 14 and 15 of the paper
