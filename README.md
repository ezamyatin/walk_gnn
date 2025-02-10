# WalkGNN #
This is the official codebase of the paper

**Ego-net Link Prediction with WalkGNN: A Scalable Approach for Large-Scale Social Graphs**

Evgeny Zamyatin

## Dataset ##
The **EgoVK** dataset is avalilable [here](https://cloud.mail.ru/public/XkJG/e7JnntX7H).

The **Yeast** dataset is avalilable [here](https://www.chrsmrrs.com/graphkerneldatasets/YEAST.zip).


## Reproduction ##
To reproduce the results, use the following command.

Train:
```bash
python3 train.py --model walk_gnn --device cuda:0
```

Validate:
```bash
python3 validate.py --model walk_gnn --device cuda:0 --state_dict_path <path>
```
