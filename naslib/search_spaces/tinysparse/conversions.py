from torch.nn.modules import module
from naslib.search_spaces.core.primitives import Identity, ReLUConvBN
from naslib.search_spaces.darts.primitives import FactorizedReduce
from naslib.search_spaces.tinysparse.primitives import DestroySignal


OPS = [
    'id',
    'conv3x3',
    'conv5x5',
    'min',
    'max',
    'mean',
    'var',
    'zero',
    'ones'
]

def convert_naslib_to_compact(graph):
    n_nodes = len(graph.nodes)
    cells = []
    compact = []

    for i in range(1, n_nodes+1):
        if 'subgraph' in graph.nodes[i] and 'cell_' in graph.nodes[i]['subgraph'].name:
            cells.append(graph.nodes[i]['subgraph'])

    for cell in cells:
        op = cell.edges[1, 2]['op']
        module_type = ''
        if isinstance(op, DestroySignal):
            module_type = op.module_type
        elif isinstance(op, Identity) or isinstance(op, FactorizedReduce):
            module_type = 'id'
        elif isinstance(op, ReLUConvBN):
            kernel_size = op.kernel_size
            module_type = f'conv{kernel_size}x{kernel_size}'
        else:
            raise NotImplementedError(f'Operator of type {type(op)} not recognized for conversion to string representation.')

        compact.append(module_type)

    return compact

