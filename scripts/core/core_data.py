#!/usr/bin/python
# -*- coding: utf-8 -*-

from six import add_metaclass
from abc import ABCMeta

Altitude = None
Road = None
Crosswalk = None
Light = None
Sign = None


class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


@add_metaclass(ABCMeta)
class Singleton():
    
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "is_initted"):
            self._init_module(*args, **kwargs)
            self.is_initted = True
    
    def _init_module(self):
        pass


