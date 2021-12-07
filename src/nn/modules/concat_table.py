
import torch

from nn.modules.container import Container


class ConcatTable(Container):

    def __init__(self, ):
        super(ConcatTable, self).__init__()
        self.modules = []
        self.output = []

    def update_output(self, input):
        self.output = [module.update_output(input) for module in self.modules]
        return self.output

    def update_grad_input(self, input, gradOutput):
        return self._backward('update_grad_input', input, gradOutput)

    def backward(self, input, gradOutput, scale=1):
        return self._backward('backward', input, gradOutput, scale)

    def _backward(self, method, input, gradOutput, scale=1):
        is_table = isinstance(input, list)
        was_table = isinstance(self.gradInput, list)
        if is_table:
            for i, module in enumerate(self.modules):
                if method == 'update_grad_input':
                    current_grad_input = module.update_grad_input(input, gradOutput[i])
                elif method == 'backward':
                    current_grad_input = module.backward(input, gradOutput[i], scale)
                else:
                    assert False, "unknown target method"
                if not isinstance(current_grad_input, list):
                    raise RuntimeError("currentGradInput is not a table!")

                if len(input) != len(current_grad_input):
                    raise RuntimeError("table size mismatch")

                if i == 0:
                    self.gradInput = self.gradInput if was_table else []

                    def fn(l, i, v):
                        if i >= len(l):
                            assert len(l) == i
                            l.append(v.clone())
                        else:
                            l[i].resize_as_(v)
                            l[i].copy_(v)
                    self._map_list(self.gradInput, current_grad_input, fn)
                else:
                    def fn(l, i, v):
                        if i < len(l):
                            l[i].add_(v)
                        else:
                            assert len(l) == i
                            l.append(v.clone())
                    self._map_list(self.gradInput, current_grad_input, fn)
        else:
            self.gradInput = self.gradInput if not was_table else input.clone()
            for i, module in enumerate(self.modules):
                if method == 'update_grad_input':
                    current_grad_input = module.updateGradInput(input, gradOutput[i])
                elif method == 'backward':
                    current_grad_input = module.backward(input, gradOutput[i], scale)
                else:
                    assert False, "unknown target method"
                if i == 0:
                    self.gradInput.resize_as_(current_grad_input).copy_(current_grad_input)
                else:
                    self.gradInput.add_(current_grad_input)

        return self.gradInput

    def _map_list(self, l1, l2, f):
        for i, v in enumerate(l2):
            if isinstance(v, list):
                res = self._map_list(l1[i] if i < len(l1) else [], v, f)
                if i >= len(l1):
                    assert i == len(l1)
                    l1.append(res)
                else:
                    l1[i] = res
            else:
                f(l1, i, v)
        for i in range(len(l1) - 1, len(l2) - 1, -1):
            del l1[i]
        return l1

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   +. -> '
        res = torch.typename(self)
        res = res + ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
            if i == len(self.modules) - 1:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + ext)

        res = res + line + tab + last + 'output'
        res = res + line + '}'
        return res
