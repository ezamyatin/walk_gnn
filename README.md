# WalkGNN #
This is the official codebase of the paper

**GNN Applied to Ego-nets for Friend Suggestions**

Evgeny Zamyatin

## Dataset ##
The **EgoVK** dataset is avalilable [here](https://cloud.mail.ru/public/XkJG/e7JnntX7H).


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
