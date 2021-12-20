from copy import deepcopy
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear

from torch.nn.modules.pooling import AdaptiveMaxPool2d
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.primitives import Identity, ReLUConvBN, Stem, Sequential
from naslib.search_spaces.darts.primitives import FactorizedReduce
from naslib.search_spaces.tinysparse.primitives import DestroySignal


class TinySparseSearchSpace(Graph):

    OPTIMIZER_SCOPE = [
        'n_stage_1',
        'r_stage_1',
        'n_stage_2',
        'r_stage_2',
        'n_stage_3'
    ]

    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self._create_macro_graph()

    def _create_macro_graph(self):
        self.add_nodes_from(range(1, 8))
        normal_cell = self._create_cell()

        stages = ['n_stage_1', 'n_stage_2', 'r_stage_1', 'n_stage_3', 'r_stage_2']
        self.channels = [16, 16, 32, 32, 64]

        for i, stage in zip(range(2, 8), stages):
            cell = deepcopy(normal_cell)
            cell.name = f'cell_{i}_normal' if 'n_stage' in stage else f'cell_{i}_reduction'
            self.nodes[i]['subgraph'] = cell.copy().set_scope(stage).set_input([i-1])

        for i in range(1, 8):
            self.add_edge(i, i+1)

        # ---- End of defining graph structure ---- #

        # ---- Now, to put the operations on the graph edges --- #
        print(self.edges)
        self.edges[1, 2].set('op', Stem(self.channels[0]))

        # Put the operations on the cell edges
        C_in = self.channels[0]

        for C_out, stage in zip(self.channels, stages):
            stride = 1 if 'n_stage' in stage else 2
            self.update_edges(
                update_func=lambda edge: self._set_cell_op(edge, C_in, C_out, stride),
                scope=stage,
                private_edge_data=True
            )
            C_in = C_out

        # Linear layer on the last edge of the macro graph
        self.edges[7, 8].set(
            'op',
            Sequential(
                AdaptiveMaxPool2d(1),
                Flatten(),
                Linear(self.channels[-1], self.num_classes),
            )
        )

    def _set_cell_op(self, edge, C_in, C_out, stride):
        edge.data.set(
            'op',
            [
                Identity() if stride == 1 else FactorizedReduce(C_in, C_out, stride, affine=False),
                ReLUConvBN(C_in, C_out, 3, padding=1, stride=stride),
                ReLUConvBN(C_in, C_out, 5, padding=2, stride=stride),
                DestroySignal(C_out, 'max', stride==2),
                DestroySignal(C_out, 'min', stride==2),
                DestroySignal(C_out, 'mean', stride==2),
                DestroySignal(C_out, 'zero', stride==2),
                DestroySignal(C_out, 'ones', stride==2),
                DestroySignal(C_out, 'noise', stride==2)
            ]
        )

    def _create_cell(self):
        cell = Graph()
        cell.name = 'cell'

        cell.add_nodes_from([1, 2])
        cell.add_edge(1, 2)
    
        return cell
