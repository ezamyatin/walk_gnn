from collections import OrderedDict

import torch

from models.gat import GATModel
from models.gin import GINEModel, GINModel
from models.heuristic import AdamicAdar, WeightedAdamicAdar
from models.ppgn import PPGN
from models.walk_gnn import WalkGNN


def get_model(args):
    if args.model == 'walk_gnn':
        model = WalkGNN(node_dim=74, edge_dim=3, hid_dim=8, num_blocks=6, mlp_layers=2)
    elif args.model == 'walk_gnn_2b':
        model = WalkGNN(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=2, mlp_layers=2)
    elif args.model == 'walk_gnn_4b':
        model = WalkGNN(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=4, mlp_layers=2)
    elif args.model == 'walk_gnn_8b':
        model = WalkGNN(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=8, mlp_layers=2)
    elif args.model == 'walk_gnn_no_edge_attr':
        model = WalkGNN(node_dim=8, edge_dim=None, hid_dim=8, num_blocks=6, mlp_layers=2)
    elif args.model == 'walk_gnn_no_node_attr':
        model = WalkGNN(node_dim=None, edge_dim=4, hid_dim=8, num_blocks=6, mlp_layers=2)
    elif args.model == 'walk_gnn_no_attr':
        model = WalkGNN(node_dim=None, edge_dim=None, hid_dim=8, num_blocks=6, mlp_layers=2)
    elif args.model == 'gine':
        model = GINEModel(node_dim=8, edge_dim=4, hid_dim=256, num_blocks=6)
    elif args.model == 'rgine':
        model = GINEModel(node_dim=8, edge_dim=4, hid_dim=256, num_blocks=6, r_version=True)
    elif args.model == 'gine_ohe':
        model = GINEModel(node_dim=8, edge_dim=4, hid_dim=256, num_blocks=6, use_degree_ohe=True, max_nodes=300)
    elif args.model == 'gine_id_ohe':
        model = GINEModel(node_dim=8, edge_dim=4, hid_dim=256, num_blocks=6, use_id_ohe=True, max_nodes=300)
    elif args.model == 'gin_ohe':
        model = GINModel(hid_dim=256, num_blocks=6, use_degree_ohe=True, max_nodes=300)
    elif args.model == 'gin_constant':
        model = GINModel(hid_dim=256, num_blocks=6, use_degree_ohe=False)
    elif args.model == 'gat':
        model = GATModel(node_dim=8, edge_dim=4, hid_dim=256, num_blocks=6)
    elif args.model == 'aa':
        model = AdamicAdar()
    elif args.model == 'waa':
        model = WeightedAdamicAdar()
    elif args.model == 'ppgn':
        model = PPGN(node_dim=8, edge_dim=4, hid_dim=32, num_blocks=6, mlp_layers=3)
    elif args.model == 'ppgn_no_attr':
        model = PPGN(node_dim=8, edge_dim=4, hid_dim=32, num_blocks=6, mlp_layers=3, ignore_attr=True)
    else:
        assert False

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
        except:
            state_dict = torch.load(args.state_dict_path)
            model.load_state_dict(OrderedDict(zip(map(lambda e: e[len('model.'):], state_dict.keys()), state_dict.values())))

    if args.device is not None:
        model.to(args.device)

    return model
