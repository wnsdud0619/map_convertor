#! /usr/bin/python
# -*- coding: utf-8 -*-

import six
from abc import (
    ABCMeta,
    abstractmethod,
)

from scripts.core.core_data import (
    Singleton,
)
from scripts.shape.shape_data import (
    Shape,
    Convert,
)
from scripts.shape.shape_file import (
    SHP_Loader,
    SHP_Saver,
)
from scripts.functions.coordinate_functions import (
    get_point_on_points,
)
from scripts.functions.print_functions import (
    Process_Counter,
    class_print,
    log_print,
    warning_print,
)

# --------------------------------------------------

@six.add_metaclass(ABCMeta)
class Module(Singleton):

    def _init_module(self):
        pass

    # --------------------------------------------------

    @classmethod
    def load(cls, base_path, source_path):

        class_print("Load file")

        type_path = "shape"
        class_path = cls.__name__
        file_path = "{0}/DB/{1}/{2}/{3}.db".format(base_path, type_path, source_path, class_path)

        if SHP_Loader.load(file_path):
            if cls.__name__ == "Shape_Generator":
                Shape.add_interface()
            return True
        return False

    @classmethod
    def save(cls, base_path, source_path):
        
        class_print("Save file")

        type_path = "shape"
        class_path = cls.__name__
        file_path = "{0}/DB/{1}/{2}/{3}.db".format(base_path, type_path, source_path, class_path)

        if SHP_Saver.save(file_path):
            return True
        return False

    @classmethod
    def execute(cls, *args, **kwargs):
        
        def save_shape():

            domain = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C3", "C4", "C5", "C6"]
            counter = Process_Counter(len(domain))

            if cls.__name__ not in ["Shape_Loader", "Shape_Reviser"]:
                domain = [x + "_POST" for x in domain]

            for shape_type in domain:
                map = Convert.create_map()
                for shape in Shape.get_shape_datas(shape_type).values():
                    line3d = Convert.convert_to_lineString3d(shape.points)
                    if shape_type in ["A2", "A2_POST"]:
                        line3d.attributes["ID"] = str(shape.ID)
                        line3d.attributes["FromNode"] = str(shape.FromNodeID)
                        line3d.attributes["ToNode"] = str(shape.ToNodeID)
                        if shape_type == "A2":
                            _link = Shape.get_shape("A2", shape.R_LinkID)
                        else:
                            _link = Shape.get_post("A2", shape.R_LinkID)
                        if _link != None:
                            _line = [
                                get_point_on_points(shape.points, division=2),
                                get_point_on_points(_link.points, division=2)
                            ]
                            _line3d = Convert.convert_to_lineString3d(_line)
                            map.add(_line3d)

                    map.add(line3d)
                Convert.save_map(cls.__name__, "{0}.osm".format(shape_type), map, sub_dir="result")
                counter.add()
                counter.print_sequence("[{0}] Save shape ({1})".format(cls.__name__, shape_type))
            counter.print_result("[{0}] Save shape".format(cls.__name__, shape_type))

        class_print("{0}".format(cls.__name__))

        # 1. Module 기능 수행
        if not cls().do_process(*args, **kwargs):
            return False
            
        save_shape()
        
        # 2. Module 단계 결과 save
        if kwargs.get("save_flag"):
            return cls.save(kwargs.get("base_path"), kwargs.get("source_path"))
        else:
            return True

    @abstractmethod
    def do_process(self, *args, **kwargs):
        pass
