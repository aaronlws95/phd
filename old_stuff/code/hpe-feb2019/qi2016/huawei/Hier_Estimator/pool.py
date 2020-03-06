#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy
class RealtimeHandposePipeline(object):

    def __init__(self,a):
        self.a=a
    def func(self,a, b,c):
        print(a,b,c)
        return a + b+c

def main():

    a=numpy.ones((2,3,3))
    tmp=RealtimeHandposePipeline(a)
    b=numpy.ones((2,3,3))
    c=1
    a_args = [a,a,a,a]
    second_arg = a
    with Pool() as pool:
        L=pool.starmap(tmp.func, [(a, b,c), (a, b,c), (a, b,c)])
        print(L)
        # M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        # N = pool.map(partial(func, b=second_arg), a_args)
        # print(L,M,N)
        # assert L == M == N

if __name__=="__main__":
    freeze_support()
    main()