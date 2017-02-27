# EVE: Explainable Vector Based Embedding Technique Using Wikipedia
## Authors of Research contributions: Muhammad Atif Qureshi, Derek Greene

###Tested: Python 3.4.3, Anaconda3 Jupyter Notebook 4.2.0

####Steps

0. From src, run 

		python setup.py

 > This will create empty directory structure i.e., dataset and output directories

1. Download dataset files from <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/V7MDP6> and extract them into the folder *dataset*

 >   After extraction, the structure shall look like  **dataset/tasks/** and **dataset/indexes/**

2. From src, run (Experiment 1: intrusion detection), 

		python intrusion_task.py

 >  It will reproduce results as shown in Table 4 and explanations as shown in Table 5 and 6 of the paper

3. Run Notebook from the root folder and execute ipython/intrusion_p-value.ipynb

 >  It will reproduce p-values as discussed in Table 4 of the paper

4. From src, run (Experiment 2: ability to cluster)

		python ability_to_cluster_pairwise_similarity.py

 >  It will calculate pairwise scores which is used in later script and it will reproduce explanations as shown in Table 10 and 11 of the paper

5. Run Notebook from the root folder and execute ipython/cluster-validation-measures.ipynb

 > It will reproduce results as shown Table 7, 8, and 9 of the paper

6. Run Notebook from the root folder and execute ipython/cluster-visualisations.ipynb

 > It will reproduce visualisation as shown in Figure 4 of the paper.

7. From src, run (Experiment 3: sorting relevant items first)

		python sorting_relevant_items_first_task.py

 > It will reproduce results as shown in Tables 12 and 13, and explanations as shown in Table 14 and 15 of the paper
