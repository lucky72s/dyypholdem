
from nn.modules.module import Module
from nn.modules.utils import clear


class Container(Module):

    def __init__(self, *args):
        super(Container, self).__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)
        return self

    def get(self, index):
        return self.modules[index]

    def size(self):
        return len(self.modules)

    def apply_to_modules(self, func):
        for module in self.modules:
            func(module)

    def zero_grad_parameters(self):
        self.apply_to_modules(lambda m: m.zeroGradParameters())

    def update_parameters(self, learning_rate):
        self.apply_to_modules(lambda m: m.update_parameters(learning_rate))

    def training(self):
        self.apply_to_modules(lambda m: m.training())
        super(Container, self).training()

    def evaluate(self, ):
        self.apply_to_modules(lambda m: m.evaluate())
        super(Container, self).evaluate()

    def reset(self, stdv=None):
        self.apply_to_modules(lambda m: m.reset(stdv))

    def parameters(self):
        w = []
        gw = []
        for module in self.modules:
            mparam = module.parameters()
            if mparam is not None:
                w.extend(mparam[0])
                gw.extend(mparam[1])
        if not w:
            return
        return w, gw

    def clear_state(self):
        clear('output')
        clear('gradInput')
        for module in self.modules:
            module.clear_state()
        return self
