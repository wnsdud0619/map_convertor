#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
from functools import reduce

import shapefile

from scripts.shape.shape_data import (
    Shape,
)
from scripts.shape.shape_module import (
    Module,
)
from scripts.shape.shape_data import (
    Shape,
)
from scripts.functions.coordinate_functions import (
    convert_latlon_to_utm,
)
from scripts.functions.print_functions import (
    Process_Counter,
    Color,
    log_print,
    warning_print,
)


class Shape_Loader(Module):

    file_data = {
        "A1" : "A1_NODE",
        "A2" : "A2_LINK",
        "A3" : "A3_DRIVEWAYSECTION",
        "A4" : "A4_SUBSIDLARYSECTION",
        "A5" : "A5_PARKINGLOT",
        "B1" : "B1_SAFETYSIGN",
        "B2" : "B2_SURFACELINEMARK",
        "B3" : "B3_SURFACEMARK",
        "C1" : "C1_TRAFFICLIGHT",
        "C2" : "C2_KILOPOST",
        "C3" : "C3_VEHICLEPROTECTIONSAFETY",
        "C4" : "C4_SPEEDBUMP",
        "C5" : "C5_HEIGHTBARRIER",
        "C6" : "C6_POSTPOINT",
    }

    def load_shape(self, dir_path, coordinate_type="utm", encoding_type="ISO-8859-1"):
        """
        Load shapefile
        """ 

        def check_directory(dir_path):
            if not os.path.isdir(dir_path):            
                warning_print("Directory does not exist ({0})".format(dir_path))
                return False
            return True    

        def get_name(shape_type):
            return Shape_Loader.file_data.get(shape_type)

        def create_shapeReader(dir_path, shape_type, encoding_type):
            try:
                file_name = get_name(shape_type)
                shape_reader = shapefile.Reader("{0}/{1}".format(dir_path, file_name), encoding=encoding_type)
            except shapefile.ShapefileException as e:
                warning_print("Failed to load shape ({0}) => No such file ({1})".format(shape_type, Color.highlight(file_name)))
                return None
            else:
                return shape_reader

        def check_columns(shape_type):
            if Shape.get_columns(shape_type) == None:
                return False
            return True

        def extract_record(shape_type, record):

            instance = Shape.Instance()

            # 1. SHP column 명 추출
            keys = record.as_dict().keys()
            # 2. 소문자 변환
            _keys = [key.lower() for key in keys]
            # 3. Shape 클래스에 정의된 column 와 대조 
            columns = Shape.get_columns(shape_type)
            for column in columns[:-1]:
                # 1) Shape 클래스에 정의된 column 과 일치하는 항목 추출
                key = keys[_keys.index(column.lower())]
                # 2) 항목의 값 추출
                field = getattr(record, key)
                # 3) 정의된 형식으로 변환
                value = Shape.convert_field(column, field)
                # 4) 인스턴스에 등록
                instance.replace(**{column : value})

            return instance

        def extract_shape(shape, coordinate_type):

            points = []

            if len(shape.points) > 0:
                # 1. 좌표계 변환            
                shape_points = shape.points if coordinate_type == "utm" else convert_latlon_to_utm(shape.points)
                # 2. 고도값(z) 연계
                for index, (x, y) in enumerate(shape_points):
                    # - 고도값 추출
                    alt = shape.z[index]
                    # - 좌표 생성
                    point = (x, y, alt)
                    # - 목록 추가
                    points.append(point)

            return points

        # - Directory 체크
        if not check_directory(dir_path):
            return False

        domain = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C3", "C4", "C5", "C6"]
        
        # - domain 에서 각 shape 유형 추출
        for shape_type in domain:

            if not check_columns(shape_type):
                warning_print("Failed to load shape ({0}) => No matching columns".format(shape_type))
                continue

            # - shape reader 생성
            shape_reader = create_shapeReader(dir_path, shape_type, encoding_type)

            # - file_path 오류 발생 시 해당 shape 유형 load 생략
            if shape_reader == None:
                continue

            counter = Process_Counter(len(shape_reader.shapeRecords()))

            # - shape reader 에서 각 shape 인스턴스 추출
            for shape_record in shape_reader.shapeRecords():
                # 1) record 데이터 추출
                instance = extract_record(shape_type, shape_record.record)
                # 2) shape 데이터 추출
                points = extract_shape(shape_record.shape, coordinate_type)
                # 3) record + shape
                instance.replace(points=points)
                # - 좌표 데이터가 없는 경우 제외
                if len(instance.points) > 0:
                    Shape.set_shape(shape_type, instance)
                else:
                    counter.add(item="warn")
                counter.add()
                counter.print_sequence("[{0}] Load".format(shape_type))
            counter.print_result("[{0}] Load".format(shape_type))

        return True

    def do_process(self, *args, **kwargs):

        dir_path = kwargs.get("base_path") + "/SHP/" + kwargs.get("source_path")
        
        return self.load_shape(dir_path)
