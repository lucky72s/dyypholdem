import math
import random as rng

import torch

import settings.arguments as arguments


# coefficients for MT19937
(w, n, m, r) = (32, 624, 397, 31)
a = 0x9908B0DF
(u, d) = (11, 0xFFFFFFFF)
(s, b) = (7, 0x9D2C5680)
(t, c) = (15, 0xEFC60000)
l = 18
f = 1812433253

# make an array to store the state of the generator
MT = [0 for i in range(n)]
index = n+1
lower_mask = 0x7FFFFFFF # (1 << r) - 1 // That is, the binary number of r 1's
upper_mask = 0x80000000 # lowest w bits of (not lower_mask)


# initialize the generator from a seed
def manual_seed(seed):
    # global index
    # index = n
    MT[0] = seed
    for i in range(1, n):
        temp = f * (MT[i-1] ^ (MT[i-1] >> (w-2))) + i
        MT[i] = temp & 0xffffffff


# Extract a tempered value based on MT[index]
# calling twist() every n numbers
def extract_number():
    global index
    if index >= n:
        twist()
        index = 0

    y = MT[index]
    y = y ^ ((y >> u) & d)
    y = y ^ ((y << s) & b)
    y = y ^ ((y << t) & c)
    y = y ^ (y >> l)

    index += 1
    return y & 0xffffffff


# Generate the next n values from the series x_i
def twist():
    for i in range(0, n):
        x = (MT[i] & upper_mask) + (MT[(i+1) % n] & lower_mask)
        xA = x >> 1
        if (x % 2) != 0:
            xA = xA ^ a
        MT[i] = MT[(i + m) % n] ^ xA


# return a single random number in the range 0, 1
def random():
    if arguments.use_pseudo_random:
        return extract_number() / (2 ** 32)
    else:
        return torch.rand(1).item()


# return tensor filled with size random elements
def rand(size):
    if arguments.use_pseudo_random:
        ret = arguments.Tensor(size)
        for i in range(size):
            ret[i] = random()
    else:
        ret = torch.rand(size, device=arguments.device)
    return ret


# return a single random integer in the range 0, 1
def randint(low, high):
    if arguments.use_pseudo_random:
        val = random()
        val *= (high - low + 1)
        val = math.floor(val) + low
    else:
        val = rng.randint(low, high)
    return int(val)


def uniform(size, low=0.0, high=1.0):
    ret = arguments.Tensor(size).fill_(0.0)
    for i in range(size):
        rnd = random()
        rnd = (rnd * (high - low)) + low
        ret[i] = rnd
    return ret


def uniform_(src: torch.Tensor, low=0.0, high=1.0):
    src.fill_(0.0)
    vw = src.view(-1)
    for i in range(len(vw)):
        rnd = random()
        rnd = (rnd * (high - low)) + low
        vw[i] = rnd


if __name__ == '__main__':
    manual_seed(123)
    arguments.logger.info(f"Extracted number: {extract_number()}")
