
from nn.modules.module import Module
from nn.modules.utils import clear, recursive_copy


class SelectTable(Module):

    def __init__(self, index):
        super(SelectTable, self).__init__()
        self.index = index
        self.gradInput = []

    def update_output(self, input):
        # handle negative indices
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        assert len(input) > index
        self.output = input[index]
        return self.output

    def update_grad_input(self, input, gradOutput):
        # make gradInput a zeroed copy of input
        self._zero_table_copy(self.gradInput, input)
        # handle negative indices
        index = self.index if self.index >= 0 else input.size(self.dimension) + self.index
        # copy into gradInput[index] (necessary for variable sized inputs)
        assert self.gradInput[index] is not None
        recursive_copy(self.gradInput[index], gradOutput)
        return self.gradInput

    def _zero_table_copy(self, l1, l2):
        for i, v in enumerate(l2):
            if isinstance(v, list):
                if len(l1) > i:
                    l1[i] = self._zero_table_copy(l1[i], l2[i])
                else:
                    l1.append(self._zero_table_copy([], l2[i]))
            else:
                if i >= len(l1):
                    l1.append(v.new().resize_as_(v).zero_())
                else:
                    l1[i].resize_as_(v)
                    l1[i].zero_()
        del l1[len(l2):]
        return l1

    def type(self, type=None, tensorCache=None):
        del self.gradInput[:]
        if isinstance(self.output, list):
            del self.output[:]
        return super(SelectTable, self).type(type, tensorCache)

    def clear_state(self):
        clear(self, 'gradInput')

    def __repr__(self):
        return super(SelectTable, self).__repr__() + '({})'.format(self.index)


