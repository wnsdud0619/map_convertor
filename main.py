#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
from copy import deepcopy

from scripts.shape.shape_data import (
    Shape,
    Convert,
)
from scripts.shape.shape_loader import Shape_Loader
from scripts.shape.shape_reviser import Shape_Reviser
from scripts.shape.shape_parser import Shape_Parser
from scripts.shape.shape_generator import Shape_Generator
from scripts.converter.lanelet_converter import Converter
from scripts.functions.print_functions import (
    log_print,
    time_print,
)


class Input_Manager():
    
    @classmethod
    def parse(cls):

        load_level = None
        save_flag = False
        source_path = "HDMap_UTMK"

        if len(sys.argv) > 1:
            for item in sys.argv[1:]:
                try:
                    key, value = item.split(":=")
                except:
                    continue

                if key == "load":
                    load_level = value
                elif key == "save":
                    if value == "True":
                        save_flag = True
                elif key == "map":
                    source_path = value

        return load_level, save_flag, source_path


class Manager():

    class Shape_Manager():

        modules = [
            Shape_Loader,
            Shape_Reviser,
            Shape_Parser,
            Shape_Generator,
            Converter
        ]

        def _get_base_path(self):
            return os.path.dirname(os.path.abspath(__file__))

        def _get_load_module(self, load_level):
            return {
                "loader" : Shape_Loader,
                "reviser" : Shape_Reviser,
                "parser" : Shape_Parser,
                "generator" : Shape_Generator,
            }.get(load_level)

        def _load(self, load_level, source_path):

            load_module = self._get_load_module(load_level)

            # - load_level 에 해당하는 Module 있는 경우
            if load_module != None:
                # - Module Load 성공 시
                return load_module.load(self._get_base_path(), source_path)

        def execute(self, load_level, save_flag, source_path):

            def load_modules(modules, source_path):
                # 1. Load
                # - Load 성공 시
                if self._load(load_level, source_path):
                    # - Load 된 module 까지 목록에서 제외
                    while modules.pop(0) != self._get_load_module(load_level):
                        pass 

            def execute_modules(modules):
                # 2. Execute
                while True:
                    try:
                        module = modules.pop(0)
                    except IndexError:
                        break
                    else:
                        base_path = self._get_base_path()
                        if module != Converter:
                            if not module.execute(base_path=base_path, source_path=source_path, save_flag=save_flag):
                                break
                        else:
                            module.execute(base_path, source_path)

            modules = self.__class__.modules

            s_time = time.time()

            Convert(source_path)
            load_modules(modules, source_path)
            execute_modules(modules)

            e_time = time.time()

            delay = round(e_time - s_time, 1)

            time_print(delay)

    def __init__(self, convert_type):
        self.__class__ = {
            "shape" : self.Shape_Manager, 
        }.get(convert_type)


def check_integrity():

        os.system("clear")
        load_level, save_flag, source_path = Input_Manager.parse()
        # Manager("shape").execute(load_level, save_flag, source_path)

        Convert(source_path)
        base_path = os.path.dirname(os.path.abspath(__file__))

        Shape_Loader.execute(base_path=base_path, source_path=source_path, save_flag=save_flag)
        Shape_Reviser.execute(base_path=base_path, source_path=source_path, save_flag=save_flag)
        Shape_Parser.execute(base_path=base_path, source_path=source_path, save_flag=save_flag)
        Shape_Generator.execute(base_path=base_path, source_path=source_path, save_flag=save_flag)

        # for shape_type in Shape.get_domain():
        #     shape_datas = Shape.get_shape_datas(shape_type)
        #     log_print("{0} : {1}".format(shape_type, len(shape_datas)))

        _data_pack = deepcopy(Shape().data_pack)

        # Shape_Loader.load(base_path, source_path)
        # Shape_Reviser.load(base_path, source_path)
        Shape_Parser.load(base_path, source_path)
        Shape_Generator.execute(base_path=base_path, source_path=source_path, save_flag=save_flag)

        for shape_type in Shape.get_domain():
            shape_datas = Shape.get_shape_datas(shape_type)
            _shape_datas = _data_pack.get(shape_type)
            for shape_id, shape in shape_datas.items():
                _shape = _shape_datas.get(shape_id)
                for column in Shape.get_columns(shape_type):
                    _field = getattr(_shape, column)
                    field = getattr(shape, column)
                    if _field != field:
                        if column == "Length":
                            if round(_field, 5) != round(field, 5):
                                log_print("[{3}] - {0} : {1} / {2}".format(column, _field, field, shape_type))
                        else:
                            log_print("[{3}] - {0} : {1} / {2}".format(column, _field, field, shape_type))


if __name__ == '__main__':
    try:
        # check_integrity()
        os.system("clear")
        load_level, save_flag, source_path = Input_Manager.parse()
        Manager("shape").execute(load_level, save_flag, source_path)
    except KeyboardInterrupt:
        sys.exit()    



