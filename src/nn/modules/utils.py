
import torch

# tensorCache maintains a list of all tensors and storages that have been
# converted (recursively) by calls to recursiveType() and type().
# It caches conversions in order to preserve sharing semantics
# i.e. if two tensors share a common storage, then type conversion
# should preserve that.
#
# You can preserve sharing semantics across multiple networks by
# passing tensorCache between the calls to type, e.g.
#
# > tensorCache = {}
# > net1:type('torch.cuda.FloatTensor', tensorCache)
# > net2:type('torch.cuda.FloatTensor', tensorCache)
# > nn.utils.recursiveType(anotherTensor, 'torch.cuda.FloatTensor', tensorCache)


def recursive_type(param, type, tensorCache={}):
    from .criterion import Criterion
    from .module import Module
    if isinstance(param, list):
        for i, p in enumerate(param):
            param[i] = recursive_type(p, type, tensorCache)
    elif isinstance(param, Module) or isinstance(param, Criterion):
        param.type(type, tensorCache)
    elif isinstance(param, torch.Tensor):
        if param.type() != type:
            key = param._cdata
            if key in tensorCache:
                newparam = tensorCache[key]
            else:
                newparam = torch.Tensor(param.shape).type(type)
                newparam.copy_(param)
                tensorCache[key] = newparam
            param = newparam
    return param


def recursive_resize_as(t1, t2):
    if isinstance(t2, list):
        t1 = t1 if isinstance(t1, list) else [t1]
        if len(t1) < len(t2):
            t1 += [None] * (len(t2) - len(t1))
        for i, _ in enumerate(t2):
            t1[i], t2[i] = recursive_resize_as(t1[i], t2[i])
        t1 = t1[:len(t2)]
    elif isinstance(t2, torch.Tensor):
        t1 = t1 if isinstance(t1, torch.Tensor) else t2.new()
        t1.resize_as_(t2)
    else:
        raise RuntimeError("Expecting nested tensors or tables. Got " +
                           type(t1).__name__ + " and " + type(t2).__name__ + "instead")
    return t1, t2


def recursive_fill(t2, val):
    from .module import Module
    if isinstance(t2, list):
        t2 = [recursive_fill(x, val) for x in t2]
    elif isinstance(t2, Module):
        t2.fill(val)
    elif isinstance(t2, torch.Tensor):
        t2.fill_(val)
    return t2


def recursive_add(t1, val=1, t2=None):
    if t2 is None:
        t2 = val
        val = 1
    if isinstance(t2, list):
        t1 = t1 if isinstance(t1, list) else [t1]
        for i, _ in enumerate(t2):
            t1[i], t2[i] = recursive_add(t1[i], val, t2[i])
    elif isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        t1.add_(val, t2)
    else:
        raise RuntimeError("expecting nested tensors or tables. Got " +
                           type(t1).__name__ + " and " + type(t2).__name__ + " instead")
    return t1, t2


def recursive_copy(t1, t2):
    if isinstance(t2, list):
        t1 = t1 if isinstance(t1, list) else [t1]
        for i, _ in enumerate(t2):
            t1[i], t2[i] = recursive_copy(t1[i], t2[i])
    elif isinstance(t2, torch.Tensor):
        t1 = t1 if isinstance(t1, torch.Tensor) else t2.new()
        t1.resize_as_(t2).copy_(t2)
    else:
        raise RuntimeError("expecting nested tensors or tables. Got " +
                           type(t1).__name__ + " and " + type(t2).__name__ + " instead")
    return t1, t2


def contiguous_view(output, input, *args):
    if output is None:
        output = input.new()
    if input.is_contiguous():
        output.set_(input.view(*args))
    else:
        output.resize_as_(input)
        output.copy_(input)
        output.set_(output.view(*args))
    return output


# go over specified fields and clear them. accepts
# nn.clear_state(self, ['_buffer', '_buffer2']) and
# nn.clear_state(self, '_buffer', '_buffer2')
def clear(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    def _clear(f):
        if not hasattr(self, f):
            return
        attr = getattr(self, f)
        if isinstance(attr, torch.Tensor):
            attr.set_()
        elif isinstance(attr, list):
            del attr[:]
        else:
            setattr(self, f, None)
    for key in args:
        _clear(key)
    return self
