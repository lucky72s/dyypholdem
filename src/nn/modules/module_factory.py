
from nn.modules.sequential import Sequential
from nn.modules.concat_table import ConcatTable
from nn.modules.select_table import SelectTable
from nn.modules.add_table import CAddTable
from nn.modules.linear import Linear
from nn.modules.prelu import PReLU
from nn.modules.batch_norm import BatchNormalization
from nn.modules.narrow import Narrow
from nn.modules.dot_product import DotProduct
from nn.modules.replicate import Replicate
from nn.modules.mul_constant import MulConstant


class ModuleFactory(object):

    module_types = {
        "nn.Sequential": Sequential,
        "nn.ConcatTable": ConcatTable,
        "nn.SelectTable": SelectTable,
        "nn.CAddTable": CAddTable,
        "nn.Linear": Linear,
        "nn.PReLU": PReLU,
        "nn.BatchNormalization": BatchNormalization,
        "nn.Narrow": Narrow,
        "nn.DotProduct": DotProduct,
        "nn.Replicate": Replicate,
        "nn.MulConstant": MulConstant,
    }

    def create_module(self, module) -> object:
        return self.module_types[module].__new__(self.module_types[module])
