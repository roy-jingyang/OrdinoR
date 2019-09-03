#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from abc import ABC, abstractmethod
class Base(ABC):
    _a = None
    _b = None
    _c = None

    @abstractmethod
    def set_a(self):
        pass

    @abstractmethod
    def set_b(self):
        pass

    @abstractmethod
    def set_c(self):
        pass

class A(Base):
    def __init__(self, v):
        print('instantiate A')
        self.set_a(v)
        self.set_b()
        self.set_c()
    
    def set_a(self, v):
        self._a = list()
        self._a.append(v)

    def set_b(self):
        pass

    def set_c(self):
        pass

class B(Base):
    def __init__(self, v):
        print('instantiate B')
        self.set_a()
        self.set_b(v)
        self.set_c()

    def set_a(self):
        pass

    def set_b(self, v):
        self._b = list()
        self._b.append(v * -1)

    def set_c(self):
        pass

class C(A, B):
    def __init__(self, v1, v2, v3):
        print('instantiate C')
        print('---')
        A.set_a(self, v1)
        B.set_b(self, v2)
        self.set_c(v3)

    def set_c(self, v):
        self._c = list()
        self._c.append('Nah')

# List input parameters from shell


if __name__ == '__main__':
    #base = Base()
    #exit("everything's good")

    ia = A(6)
    print(ia._a)
    print(ia._b)
    print(ia._c)

    ia2 = A(7)
    print(ia2._a)
    print(ia2._b)
    print(ia2._c)

    ib = B(16)
    print(ib._a)
    print(ib._b)
    print(ib._c)
    
    #exit("everything's good")

    ic = C(7, 17, 27)
    print(ic._a)
    print(ic._b)
    print(ic._c)

    ic2 = C(107, 1017, 1027)
    print(ic2._a)
    print(ic2._b)
    print(ic2._c)


