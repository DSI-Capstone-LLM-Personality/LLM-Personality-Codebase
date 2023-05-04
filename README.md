# LLM-Personality-Codebase

Natural Language Processing (NLP) research about language model personality, supported by [Columbia Unversity Data Science Institute](https://datascience.columbia.edu/) and [JP Morgan](https://www.jpmorgan.com/global).

**Contributors**: [Xiaoyang Song](https://github.com/Xiaoyang-Song), [Kiyan Mohebbizadeh](https://github.com/kmohebbizadeh), [Shujie Hu](https://github.com/tracyhsj), [Morris Hsieh](https://github.com/MorrisHsieh3059), and [Anant Singh](https://github.com/95anantsingh).

**Mentor**: [Akshat Gupta](https://scholar.google.com/citations?user=v80j6o0AAAAJ&hl=en) from JP Morgan AI Research.

## Reproducibility

The results and analysis in this work is 100% reproducible. To run the code, select one configuration file from the `config` folder (or you can always create your own by simply following the YAML file format). Below is an example running Constraint-Search MPI experiments on BERT-Base to examine order symmetry with no index.

```
python main_mpi.py --config=config/Constraint/order-symmetry/BERT-Base/non-index.yaml
```

Use `-seed` to set the seed for reproducibility. The default value is `2023`. Note that all experimental results (except for the one with order shuffling) are obtained under the default seed. For order-symmetry-related experiments, please specify the seed in the configuration file.

```
python main_mpi.py --config=config/Constraint/order-symmetry/BERT-Base/non-index.yaml --seed=<your seed here>
```

Use `-tag` to add a special identifier for the output logging files and checkpoints. Default is empty string.

```
python main_mpi.py --config=config/Constraint/order-symmetry/BERT-Base/non-index.yaml --tag=<your tag here>
```

Use `-verbose` to specify whether you want to see the detailed output. Default is `False`.

```
python main_mpi.py --config=config/Constraint/order-symmetry/BERT-Base/non-index.yaml --verbose
```

Note that if in any case you meet problems like `FileNotFoundError`, be sure to double check that you follow the structure of this repository and the current working directory is correct. The following command maybe helpful.

```
export PYTHONPATH=$PATHONPATH:`pwd`
```

### Running Order-Symmetry Experiment

To run order-symmetry experiment efficiently, simple type in your terminal the following commands:

```
time bash run.sh -r <regime> -t <type> -m <model> -d <*.yaml config file>
```

For example, to run the _Constraint_ search _order-symmetry_ experiment on _BERT-Base_ model with _non-indexed_ template, simply type in:

```
time bash run.sh -r 'Constraint' -t 'order-symmetry' -m 'BERT-Base' -d 'non-index.yaml'
```

For more details of file structures, please check the `config` folder for details. Note that You can also directly run `time bash run.sh` without providing those command line arguments. Then you will follow the instructions to enter everything needed. In addition, the default verbosity is `False`. To change this, please modify `run.sh` according to instructions at line `75`.

### Available Prompt Templates

In this section, we introduce the candidates of templates which we will perform template selection on. To summarize, we basically borrow three different templates from the [MPI paper](https://arxiv.org/abs/2206.07550). Please check [this file](./template/candidates.md) for details of candidate templates generation.

To run template selection on our candidate templates, first put all candidates into the folder `template/candidates/` and then change the configuration in `config/template-selection/template-selection.yaml` as you want. Then type in the following command to run:

```
python3 template/selection.py --config=config/template-selection/template-selection.yaml
```

After running the experiment, go to `checkpoint/log/template-selection/` and find the folder name with your model family and version. Inside the folder, the results are recorded in `scores.txt` files. Note that the score calculated is mutual information for Close Vocabulary methods and percentage of valid answers for Open Vocabulary method.
