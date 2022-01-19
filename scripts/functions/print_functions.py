#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time
import inspect
from functools import (
    reduce,
)


class Logger():

    level = 1

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


class Color():
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DEFAULT = '\33[37m'

    @classmethod
    def set_color(cls, msg, color):
        return color + msg + cls.ENDC

    @classmethod
    def highlight(cls, msg, color=WARNING):
        return color + str(msg) + cls.ENDC


class Process_Counter():

    def __init__(self, total, base=["error", "warn", "fix", "executed"]):
        self.base = base
        self.total = total
        for item in self.base:
            setattr(self, item, 0)
        self.__start_clock()

    def __start_clock(self):
        self.s_time = time.time()

    def __end_clock(self):
        self.e_time = time.time()
        
    def __percent_print(self, msg, is_done=False, level=Logger.INFO):

        def create_box(item, is_done):
            count = getattr(self, item)
            box_color = {
                "error" : Color.FAIL, 
                "warn" : Color.WARNING,
                "fix" : Color.OKCYAN, 
                "executed" : Color.OKGREEN if count >= getattr(self, "total") else Color.WARNING,
            }.get(item)
            
            front = "{0} = ".format(item)
            rear = "{0: >8}".format(count) 

            if item == "executed" and is_done:
                content = "{0: <23}".format(Color.set_color(front + rear, box_color)) 
            else:
                content = (
                    "{0: <14}".format(front + rear) if count < 1 else 
                    "{0: <23}".format(Color.set_color(front + rear, box_color)) 
                )

            box = "[ " + content
            box += " / {0: >6} ]".format(getattr(self, "total")) if item == "executed" else " ]"

            return box

        if Logger.level > level:
            return

        body = "  - {0: <50} : ".format(msg) 

        table = reduce(lambda x, y : x + y, [create_box(item, is_done) for item in self.base])
            
        if not is_done:
            if (self.total) > 0:
                percent = round((float(getattr(self, "executed"))) / float((getattr(self, "total"))) * 100.0, 1)
            else:
                percent = 0.0
            result = " : ({0} %)".format(percent)
        else:
            result = " : ({0: <5} sec)\n".format(round(self.e_time - self.s_time, 1))

        overlap_print(body + table + result)

    def add(self, item="executed", count=1):
        setattr(self, item, getattr(self, item) + count) 

    def print_sequence(self, msg):
        self.__percent_print(msg)

    def print_result(self, msg):
        self.__end_clock()
        self.__percent_print(msg, is_done=True)


def overlap_print(msg):
    sys.stdout.write("\033[F")  
    sys.stdout.write("\033[K")
    print(msg)

def log_print(msg, depth=0, level=Logger.DEBUG, color=Color.DEFAULT, is_overlap=False, is_highlight=False):

    if Logger.level > level:
        return

    method_name = inspect.stack()[1 + depth][3] 

    head = "  -> [ {0: <30} ] : ".format(Color.set_color(method_name, color))

    body = "{0: <50} ".format(msg) if not is_highlight else "{0: <59} ".format(Color.highlight(msg))

    sentence = head + body + ("\n" if not is_overlap else "")

    overlap_print(sentence)

def warning_print(*args, **kwargs):

    data = {
        "level" : Logger.WARNING,
        "color" : Color.WARNING,
        "depth" : 1,
    }

    for key, value in data.items():
        if kwargs.get(key) == None:
            kwargs[key] = value

    log_print(*args, **kwargs)

def error_print(e, color=Color.FAIL, level=Logger.ERROR, depth=0):

    if Logger.level > level:
        return

    head = " {0: <40} ] : ".format((">") + " [ " + Color.set_color(inspect.stack()[1 + depth][3], color))

    file_name = inspect.getmodule(inspect.stack()[1][0]).__file__
    body = "{0: <50} ({1} - Line = {2})".format(str(e), file_name.split("/")[-1].split(".")[0], str(inspect.stack()[1 + depth][2]))

    sentence = head + body + ("\n")

    overlap_print(sentence)

def class_print(msg, level=Logger.INFO, color=Color.WARNING):

    if Logger.level > level:
        return

    overlap_print("{0:=<180}\n".format(" [ {0} ] ".format("".join([color, Color.BOLD, msg, Color.ENDC, Color.ENDC]))))

def chapter_print(msg, level=Logger.INFO, color=Color.DEFAULT):

    if Logger.level > level:
        return

    overlap_print("  {0:-<170}\n".format("[ {0} ] ".format("".join([color, msg, Color.ENDC]))))

def time_print(delay, level=Logger.INFO, color=Color.OKCYAN):
    
    if Logger.level > level:
        return

    msg = "Delay = {0}".format(delay)

    overlap_print("{0:=<180}\n".format(" [ {0} ] ".format("".join([color, Color.BOLD, msg, Color.ENDC, Color.ENDC]))))

def interrupt_print(*args, **kwargs):

    class_print("Interrupt", color=Color.FAIL)

    data = {
        "level" : Logger.WARNING,
        "color" : Color.WARNING,
        "depth" : 1,
    }

    for key, value in data.items():
        if kwargs.get(key) == None:
            kwargs[key] = value

    log_print(*args, **kwargs)
    sys.exit()