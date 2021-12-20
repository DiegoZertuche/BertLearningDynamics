# BertLearningDynamics

## Authors
---

* Diego Zertuche
* Benjamin Liu

## Paper
---
[How Does BERT Learn?](https://github.com/DiegoZertuche/BertLearningDynamics/blob/main/How%20Does%20Bert%20Learn.pdf)

## Our Work
---

Large language models that trained on large cor- pora have been shown to be very useful for natural language processing tasks. Given the size of these models, understanding where and when the infor- mation is obtained has become a subject of study. Recent work applied several probes to intermediate training stages to observe the developmental pro- cess of a large-scale model (Chiang et al., 2020). Building upon this, we probe the intermediate train- ing stages adding a layer by layer analysis to the probes, to better understand where in the model architecture different information is contained. We also build upon (Gururangan et al., 2020) work, by performing task adaptation pretraining to the inter- mediate training checkpoint models to understand how different levels of pretraining affect the effec- tiveness of task adaptation procedures. We find that as pretraining keeps going in the BERT model, BERT learns to separate syntactic and semantic knowledge, and pushes syntactic information to the lower layers. Task adaptation procedures are effective from early stages of the pretraining of BERT, achieving similar performances in down- stream tasks as the fully trained model. Semantic and commonsense knowledge gets swapped out for the new information contained in the task adap- tation corpora when performing task adaptation, while the syntactic information is retained by the BERT model.


## Documentation
---

To replicate the experiments of our paper, we have created a repositor containing most of the code used. The code base and experiments are located in the `edge-probing`, `lama_probing`, `lama_probing_layer_level`. The folder `data` contains the data to obtain the datasets used for the experiments, with exception of the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), used for some of the linguistic tasks  and [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) datasets

Run knowledge probes:
- Only last (12th) BERT layer: follow `lama_probing/LAMAProbing.ipynb`
- Layer by layer: follow `lama_probing_layer_level/BERTnesiaProbing.ipynb`
