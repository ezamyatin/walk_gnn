# WalkGNN #
This is the official codebase of the paper

**GNN Applied to Ego-nets for Friend Suggestions**

Anonymous Author(s)

## Dataset ##
The **EgoAnon** dataset is avalilable here.
The **Yeast** dataset is avalilable [here](https://www.chrsmrrs.com/graphkerneldatasets/YEAST.zip)


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
