# BertLearningDynamics

## Authors

* Diego Zertuche (diego_zertucheserna@g.harvard.edu)
* Benjamin Liu (tliu@g.harvard.edu)

## Paper
---
[How Does BERT Learn?](https://github.com/DiegoZertuche/BertLearningDynamics/blob/main/How%20Does%20Bert%20Learn.pdf)

## Our Work
---

Large language models that trained on large cor- pora have been shown to be very useful for natural language processing tasks. Given the size of these models, understanding where and when the infor- mation is obtained has become a subject of study. Recent work applied several probes to intermediate training stages to observe the developmental pro- cess of a large-scale model (Chiang et al., 2020). Building upon this, we probe the intermediate train- ing stages adding a layer by layer analysis to the probes, to better understand where in the model architecture different information is contained. We also build upon (Gururangan et al., 2020) work, by performing task adaptation pretraining to the inter- mediate training checkpoint models to understand how different levels of pretraining affect the effec- tiveness of task adaptation procedures. 

We find that as pretraining keeps going in the BERT model, BERT learns to separate syntactic and semantic knowledge, and pushes syntactic information to the lower layers. Task adaptation procedures are effective from early stages of the pretraining of BERT, achieving similar performances in down- stream tasks as the fully trained model. Semantic and commonsense knowledge gets swapped out for the new information contained in the task adap- tation corpora when performing task adaptation, while the syntactic information is retained by the BERT model.


## Documentation
---

To replicate the experiments of our paper, we have created a repositor containing most of the code used. The code base and experiments are located in the `edge-probing`, `lama_probing`, `lama_probing_layer_level`. The folder `data` contains the data to obtain the datasets used for the experiments, with exception of [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), used for some of the linguistic tasks,  and the [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) dataset, used for the Task Adaptation experiments, as these are datasets only available upon request.

### Environment
---

You can install all dependencies using the `requirements.txt` file, or the `requirements-no-torch.txt` if working on a environment that already has s working PyTorch installation, like a Google Colab notebook.

```
pip install -r requirements.txt
```

### Linguistic Probes
---

The code base to run the linguistic probes with an edge probing format is in the folder `edge-probing`, which relies on the PyTorch library. The code is based off the `jiant` [package](https://github.com/nyu-mll/jiant), but it is condensed and adapted for our purposes. The entire module is based off the `base_model.py` code, which takes a model architecture hosted in `HuggingFace`, in this case we use the [MultiBERTs](https://huggingface.co/google) provided by Google Research, which serves as our encoder (and tokenizer). Then depending on the configuration passed for a partiular task, a classifier head that performs span extraction, scalar mixing and pooling is attached to the encoder model. The code for the classifier head is contained on the `task_model.py` module. The `main_model.py` module calls both scripts and sets up the model to be ready to perform the edge probing, and it is the module that the user will call to create the model. For example, you could create a model with a classifier head that will perform edge probing for a task with 49 different classes, that uses 2 spans, and where the encoder weights are freezed using the code:
```
from main_model import MainModel

params = {
    'input_module': 'google/multiberts-seed-4-1900k',
    'transfer_mode': 'frozen',
    'output_mode': 'mix',
    'max_layer': -1
}

task_params = {
    'span_pooling': 'attn',
    'n_spans': 2,
    'n_classes': 49,
    'single_sided': False,
    'cls_type': 'mlp',
    'dropout': 0.2
}

model = MainModel(params, task_params, is_cuda=True)
```

The `preprocessing.py` module takes a path to a particular json file and converts the examples into an edge probing format, and converted into a PyTorch Dataset object. You can create a Dataset using the code:

```
from preprocess import MyDataset

train_task_filepath = 'data/edges/dep_ewt/en_ewt-ud-train.json'
labels_filepath = 'data/edges/dep_ewt/labels.txt'
train_ds = MyDataset(train_task_filepath, train_task_filepath + '.retokenized.bert-base-uncased', labels_filepath, n_spans=2)
dev_task_filepath = 'probing/data/edges/dep_ewt/en_ewt-ud-dev.json'
dev_ds = MyDataset(dev_task_filepath, dev_task_filepath + '.retokenized.bert-base-uncased', labels_filepath, n_spans=2)
```

A working version of the linguistic probing code base that can handle TPUs can be found in the git branch `tpu`.

A working example of an experiment run can be found on the [notebook](https://github.com/DiegoZertuche/BertLearningDynamics/blob/main/edge_probing/RunLinguisticProbe.ipynb) found in the edge_probe module.


### Lama Probes
---

The Lama probes to test commonsense knowledge in the models is based off FacebookAI repository. Please refer to the original repository for instructions for running the Lama probes. A working script that runs the lama probes (layer by layer or for the full model) can be found at:

Run knowledge probes:
- Only last (12th) BERT layer: follow `lama_probing/LAMAProbing.ipynb`
- Layer by layer: follow `lama_probing_layer_level/BERTnesiaProbing.ipynb`

### Task Adaptation
---

For the task adaptation procedure, the Masked Language Modelling in the unlabeled ChemProt dataset was performed utilizing the example files in the `transformers` repository. The MLM was performed utilizing a slightly modified version of the example file in `examples/pytorch/masked_lm.py`. After performing the task adaptation procedure, the training of the classification head for the classification task using the ChemProt dataset was performed using another example off the `transformers` repository, `examples/pytorch/sequence_classification.py`. These examples were used to take advantage of the already built multicore-enabling code for TPUs, as everything was done in Google Cloud Plataform, utilizing TPUs.

### Future Work
---

We plan to keep updating this repository with the further experiments that we are running regarding task adaptation in the future.
