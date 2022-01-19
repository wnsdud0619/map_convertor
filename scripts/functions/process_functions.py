#!/usr/bin/python
# -*- coding: utf-8 -*-

import six
from abc import (ABCMeta, abstractmethod)


@six.add_metaclass(ABCMeta)
class Process():

    def __init__(self):
        self.next_process = None

    def set_next(self, process):
        self.next_process = process

    def execute(self, *args):
        new_args = self._do_process(*args)
        if self.next_process != None:
            if new_args != None:
                self.next_process.execute(new_args)
            else:
                self.next_process.execute()

    @abstractmethod
    def _do_process(self):
        pass


class Process_Handler():

    def __init__(self):
        self.process_list = []

    def add_process(self, process):
        if len(self.process_list) > 0:
            self.process_list[-1].set_next(process)
        self.process_list.append(process)

    def execute(self, *args):
        if len(self.process_list) > 0:
            self.process_list[0].execute(*args)

