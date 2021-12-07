
import torch

from nn.modules.container import Container


class Sequential(Container):

    def __len__(self):
        return len(self.modules)

    def add(self, module):
        if len(self.modules) == 0:
            self.gradInput = module.gradInput

        self.modules.append(module)
        self.output = module.output
        return self

    def insert(self, module, index):
        self.modules.insert(module, index)
        self.output = self.modules[-1].output
        self.gradInput = self.modules[0].gradInput

    def remove(self, index=-1):
        del self.modules[index]

        if len(self.modules) > 0:
            self.output = self.modules[-1].output
            self.gradInput = self.modules[0].gradInput
        else:
            self.output = torch.Tensor()
            self.gradInput = torch.Tensor()

    def update_output(self, input):
        current_output = input
        for i, module in enumerate(self.modules):
            current_output = module.update_output(current_output)
        self.output = current_output
        return self.output

    def update_grad_input(self, input, gradOutput):
        current_grad_output = gradOutput
        for prev, current in self._iter_with_prev():
            current_grad_output = current.updateGradInput(prev.output, current_grad_output)
        self.gradInput = self.modules[0].updateGradInput(input, current_grad_output)
        return self.gradInput

    def backward(self, input, gradOutput, scale=1):
        current_grad_output = gradOutput
        for prev, current in self._iter_with_prev():
            current_grad_output = current.backward(prev.output, current_grad_output, scale)
        self.gradInput = self.modules[0].backward(input, current_grad_output, scale)
        return self.gradInput

    def _iter_with_prev(self):
        return zip(self.modules[-2::-1], self.modules[-1:0:-1])

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = ' -> '
        res = 'nn.Sequential'
        res = res + ' {' + line + tab + '[input'
        for i in range(len(self.modules)):
            res = res + next + '(' + str(i) + ')'

        res = res + next + 'output]'
        for i in range(len(self.modules)):
            res = res + line + tab + '(' + str(i) + '): ' + str(self.modules[i]).replace(line, line + tab)

        res = res + line + '}'
        return res
