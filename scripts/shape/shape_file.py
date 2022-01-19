#!/usr/bin/python
# -*- coding: utf-8 -*-

from six import add_metaclass
from abc import (
    ABCMeta,
    abstractmethod,
)
from functools import (
    reduce,
)

from scripts.shape.shape_data import Shape
from scripts.file.file_data import (
    Core,
    Loader,
    Saver,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
    error_print,
)


class SHP_Core(Core):

    @add_metaclass(ABCMeta)
    class Base():
        
        def __init__(self, shape_type):
            self.shape_type = shape_type

        def _create_record_table(self, table_name, columns):
            """
            SHP 설정 테이블 생성
            """
            column_list = reduce(lambda x, y : "{0}, {1}".format(x, y), ["{0} text".format(column) for column in columns])
            SHP_Core._create_table(table_name, column_list)

        @abstractmethod
        def _init_table(self):
            pass

        # --------------------------------------------------

        @abstractmethod
        def _convert_record_field(self, field, column):
            pass

        @abstractmethod
        def _convert_record_to_shape(self, record):
            pass

        @abstractmethod
        def _load(self):
            pass

        # --------------------------------------------------

        @abstractmethod
        def _convert_shape_to_tuple(self):
            pass

        @abstractmethod
        def _convert_shape_field(self, field, column):
            pass

        @abstractmethod
        def _save(self):
            pass

    # --------------------------------------------------

    class SHP(Base):
        """
        A1 ~ C6 기본 SHP Load / Save class
        - 설정 데이터 / 좌표 데이터 
        """

        def _create_shape_table(self, table_name):
            """
            SHP 좌표 테이블 생성
            """

            column_list = "id integer not null primary key unique, x real, y real, z real"

            SHP_Core._create_table(table_name, column_list)

        def _init_table(self):

            def init_record(table_name):

                columns = Shape.get_columns(table_name)

                if not SHP_Core._check_table(table_name):
                    self._create_record_table(table_name, columns)

                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_record_table(table_name, columns)

            def init_shape(table_name):
                
                table_name = table_name + "_PTS"

                if not SHP_Core._check_table(table_name):
                    self._create_shape_table(table_name)

                columns = ["x", "y", "z"]
                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_shape_table(table_name)

            table_name = self.shape_type

            init_record(table_name)
            init_shape(table_name)

        # --------------------------------------------------

        def _convert_record_to_points(self, record):
            
            points = []

            points_field = record[-1]
            points_indices = [int(index) for index in points_field.split()]
            table_name = self.shape_type + "_PTS"
            start_index = points_indices[0]
            end_index = points_indices[-1]

            SHP_Core._execute("select * from {0} where id >= {1} and id <= {2}".format(table_name, start_index, end_index))
            points_records = SHP_Core().cursor.fetchall()

            for point_row in points_records:
                point = tuple([float(v) for v in point_row[1:]])
                points.append(point)

            return points

        def _convert_record_field(self, field, column):
            return Shape.convert_field(column, field)

        def _convert_record_to_shape(self, record, points):
            
            shape = Shape.Instance()
            columns = Shape.get_columns(self.shape_type)

            # - points column 을 제외 (마지막)
            for index in range(len(columns) - 1):
                column = columns[index]
                field = record[index]
                converted_field = self._convert_record_field(field, column)
                shape.replace(**{column : converted_field})
            else:
                shape.replace(points=points)

            shape_id = getattr(shape, columns[0])

            return shape_id, shape
                
        def _load(self):
            
            table_name = self.shape_type

            records = SHP_Core._load_records(table_name)
            for record in records:
                points = self._convert_record_to_points(record)
                shape_id, shape = self._convert_record_to_shape(record, points)
                Shape.set_shape(self.shape_type, shape, shape_id=shape_id)

        # --------------------------------------------------

        def _convert_points_to_tuple(self, points):
            
            def get_last_id(table_name):

                last_id = 0
                
                SHP_Core._execute("select * from {0} order by id desc limit 1".format(table_name))
                id_pointer = SHP_Core().cursor.fetchone()

                # - 기존 point 가 존재하는 경우                
                if id_pointer != None:
                    # - 마지막 id 추출
                    last_id = id_pointer[0] 

                return last_id

            table_name = self.shape_type + "_PTS"

            points_tuples = [(None, p[0], p[1], p[2]) for p in points]
            start_id = get_last_id(table_name) + 1

            SHP_Core._executemany("insert into {0} values(?, ?, ?, ?)".format(table_name), points_tuples)
            end_id = get_last_id(table_name) 

            points_ids = list(range(start_id, end_id + 1))
            
            return points_ids

        def _convert_shape_field(self, field, column):
            return field

        def _convert_shape_to_tuple(self, shape, points_ids):
            
            fields = []

            columns = Shape.get_columns(self.shape_type)            
            # - points column 제외 (마지막)
            for index in range(len(columns) - 1):
                column = columns[index]
                field = getattr(shape, column)
                converted_field = self._convert_shape_field(field, column)
                fields.append(converted_field)
            else:
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), points_ids) if len(points_ids) > 0 else ""
                fields.append(converted_field)

            shape_tuple = tuple(fields)

            return shape_tuple

        def _save(self):
            
            def delete_table(table_name):
                SHP_Core._delete_all(table_name)
                SHP_Core._delete_all(table_name + "_PTS")

            def insert_tuples(table_name, shape_tuples):
                columns = Shape.get_columns(self.shape_type)
                column_list = reduce(lambda x, y : "{0}, {1}".format(x, y), ["?" for _ in range(len(columns))])
                SHP_Core._executemany("insert into {0} values({1})".format(table_name, column_list), shape_tuples)

            table_name = self.shape_type

            delete_table(table_name)

            shape_tuples = []
            shape_datas = Shape.get_shape_datas(self.shape_type)
            
            for shape in shape_datas.values():
                points_ids = self._convert_points_to_tuple(shape.points)
                shape_tuple = self._convert_shape_to_tuple(shape, points_ids)
                shape_tuples.append(shape_tuple)    

            insert_tuples(table_name, shape_tuples)

    # --------------------------------------------------

    class Custom(Base):
        """
        임의 Custom SHP Load / Save class
        - 설정 데이터 (일부 좌표 데이터)
        """

        def _init_table(self):
            
            table_name = self.shape_type
            columns = Shape.get_columns(table_name)

            if not SHP_Core._check_table(table_name):
                self._create_record_table(table_name, columns)

            if not SHP_Core._check_columns(table_name, columns):
                SHP_Core._drop(table_name)
                self._create_record_table(table_name, columns)

        # --------------------------------------------------
    
        @abstractmethod
        def _convert_record_field(self, field, column):
            pass

        def _convert_record_to_shape(self, record):
            
            shape = Shape.Instance()
            columns = Shape.get_columns(self.shape_type)

            # - points column 을 제외 (마지막)
            for index in range(len(columns)):
                column = columns[index]
                field = record[index]
                converted_field = self._convert_record_field(field, column)
                shape.replace(**{column : converted_field})

            shape_id = getattr(shape, columns[0])

            return shape_id, shape
            
        def _load(self):
            
            records = SHP_Core._load_records(self.shape_type)
            for record in records:
                shape_id, shape = self._convert_record_to_shape(record)
                Shape.set_shape(self.shape_type, shape, shape_id=shape_id)

        # --------------------------------------------------

        @abstractmethod
        def _convert_shape_field(self, field, column):
            pass

        def _convert_shape_to_tuple(self, shape):
            
            fields = []

            columns = Shape.get_columns(self.shape_type)            
            for index in range(len(columns)):
                column = columns[index]
                field = getattr(shape, column)
                converted_field = self._convert_shape_field(field, column)
                fields.append(converted_field)

            shape_tuple = tuple(fields)

            return shape_tuple

        def _save(self):
            
            def delete_table(table_name):
                SHP_Core._delete_all(table_name)
                SHP_Core._delete_all(table_name + "_PTS")

            def insert_tuples(table_name, shape_tuples):
                columns = Shape.get_columns(self.shape_type)
                column_list = reduce(lambda x, y : "{0}, {1}".format(x, y), ["?" for _ in range(len(columns))])
                SHP_Core._executemany("insert into {0} values({1})".format(table_name, column_list), shape_tuples)

            table_name = self.shape_type

            delete_table(table_name)

            shape_tuples = []
            shape_datas = Shape.get_shape_datas(self.shape_type)
            
            for shape_id, shape in shape_datas.items():
                if shape_id not in [-1]:
                    shape_tuple = self._convert_shape_to_tuple(shape)
                    shape_tuples.append(shape_tuple)    

            insert_tuples(table_name, shape_tuples)

    class A2_PARENT(Custom):
        
        def _convert_record_field(self, field, column):
            """
            - child_id
            - parent_id
            """    
        
            column = column.lower()

            if field == "None":
                field = None

            if field == None:
                if column == "child_id":
                    converted_field = None
                elif column == "parent_id":
                    converted_field = None
                else:
                    converted_field = None
            else:
                if column == "child_id":
                    converted_field = str(field)
                elif column == "parent_id":
                    converted_field = str(field)
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - child_id
            - parent_id
            """

            if column == "child_id":
                converted_field = str(field)
            elif column == "parent_id":
                converted_field = str(field)
            else:
                converted_field = str(field)

            return converted_field

    class A2_CHILD(Custom):

        def _convert_record_field(self, field, column):
            """
            - parent_id
            - child_ids
            """    
        
            column = column.lower()

            if field == None:
                if column == "parent_id":
                    converted_field = None
                elif column == "child_ids":
                    converted_field = []
                else:
                    converted_field = None
            else:
                if column == "parent_id":
                    converted_field = str(field)
                elif column == "child_ids":
                    converted_field = [str(v) for v in field.split()]
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - child_id
            - parent_id
            """

            if column == "parent_id":
                converted_field = str(field)
            elif column == "child_ids":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    class A2_B2(Custom):

        def _convert_record_field(self, field, column):
            """
            - link_id
            - left_ids
            - right_ids
            """    
        
            column = column.lower()

            if field == None:
                if column == "link_id":
                    converted_field = None
                elif column == "left_ids":
                    converted_field = []
                elif column == "right_ids":
                    converted_field = []
                else:
                    converted_field = None
            else:
                if column == "link_id":
                    converted_field = str(field)
                elif column == "left_ids":
                    converted_field = [str(v) for v in field.split()]
                elif column == "right_ids":
                    converted_field = [str(v) for v in field.split()]
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - link_id
            - left_ids
            - right_ids
            """    

            if column == "link_id":
                converted_field = str(field)
            elif column == "left_ids":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            elif column == "right_ids":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    class A2P_B2T(Custom):

        def _convert_record_field(self, field, column):
            """
            - link_id
            - left_type
            - right_type
            """    
        
            column = column.lower()

            if field == None:
                if column == "link_id":
                    converted_field = None
                elif column == "left_type":
                    converted_field = 999
                elif column == "right_type":
                    converted_field = 999
                else:
                    converted_field = None
            else:
                if column == "link_id":
                    converted_field = str(field)
                elif column == "left_type":
                    converted_field = int(field)
                elif column == "right_type":
                    converted_field = int(field)
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - link_id
            - left_type
            - right_type
            """    

            if column == "link_id":
                converted_field = str(field)
            elif column == "left_type":
                converted_field = int(field)
            elif column == "right_type":
                converted_field = int(field)
            else:
                converted_field = str(field)

            return converted_field

    class A1_A2(Custom):

        def _convert_record_field(self, field, column):
            """
            - node_id
            - from_ids
            - to_ids
            """    
        
            column = column.lower()

            if field == None:
                if column == "node_id":
                    converted_field = None
                elif column == "from_ids":
                    converted_field = []
                elif column == "to_ids":
                    converted_field = []
                else:
                    converted_field = None
            else:
                if column == "node_id":
                    converted_field = str(field)
                elif column == "from_ids":
                    converted_field = [str(v) for v in field.split()]
                elif column == "to_ids":
                    converted_field = [str(v) for v in field.split()]
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - node_id
            - from_ids
            - to_ids
            """    

            if column == "node_id":
                converted_field = str(field)
            elif column == "from_ids":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            elif column == "to_ids":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    class Endpoint(Custom):

        def _convert_record_field(self, field, column):
            """
            - node_id
            - left_point
            - right_point
            """    
        
            column = column.lower()

            if field == None:
                if column == "node_id":
                    converted_field = None
                elif column == "left_point":
                    converted_field = (-1, -1, -1)
                elif column == "right_point":
                    converted_field = (-1, -1, -1)
                else:
                    converted_field = None
            else:
                if column == "node_id":
                    converted_field = str(field)
                elif column == "left_point":
                    converted_field = tuple([float(v) for v in field.split()])
                elif column == "right_point":
                    converted_field = tuple([float(v) for v in field.split()])
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - node_id
            - left_point
            - right_point
            """    

            if column == "node_id":
                converted_field = str(field)
            elif column == "left_point":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), list(field)) if len(field) > 0 else ""
            elif column == "right_point":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), list(field)) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    class Road_Bound(Base):

        def _create_shape_table(self, table_name):
            """
            SHP 좌표 테이블 생성
            """
            column_list = "id integer not null primary key unique, x real, y real, z real"

            SHP_Core._create_table(table_name, column_list)

        def _init_table(self):

            def init_record(table_name):

                columns = Shape.get_columns(table_name)

                if not SHP_Core._check_table(table_name):
                    self._create_record_table(table_name, columns)
                    return

                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_record_table(table_name, columns)

            def init_shape(table_name):
                
                table_name = table_name + "_PTS"

                if not SHP_Core._check_table(table_name):
                    self._create_shape_table(table_name)
                    return

                columns = ["x", "y", "z"]
                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_shape_table(table_name)

            table_name = self.shape_type

            init_record(table_name)
            init_shape(table_name)

        # --------------------------------------------------

        def _convert_record_to_points(self, record):
            
            left = []
            right = []
            center = []

            for index in [-3, -2, -1]:
                points_field = record[index]
                points_indices = [int(_index) for _index in points_field.split()]
                if len(points_indices) > 0:
                    table_name = self.shape_type + "_PTS"
                    start_index = points_indices[0]
                    end_index = points_indices[-1]

                    SHP_Core._execute("select * from {0} where id >= {1} and id <= {2}".format(table_name, start_index, end_index))
                    points_records = SHP_Core().cursor.fetchall()

                    for point_row in points_records:
                        point = tuple([float(v) for v in point_row[1:]])
                        side = [left, right, center][index]
                        side.append(point)

            left = None if len(left) < 1 else left
            right = None if len(right) < 1 else right
            center = None if len(center) < 1 else center

            return (left, right, center)

        def _convert_record_field(self, field, column):

            column = column.lower()

            if field == None:
                if column == "key":
                    converted_field = None
                else:
                    converted_field = None
            else:
                if column == "key":
                    converted_field = str(field)
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_record_to_shape(self, record, left, right ,center):
            
            shape = Shape.Instance()
            columns = Shape.get_columns(self.shape_type)

            # - points column 을 제외 (마지막 3개 [left, right, center] 제외)
            for index in range(len(columns) - 3):
                column = columns[index]
                field = record[index]
                converted_field = self._convert_record_field(field, column)
                shape.replace(**{column : converted_field})
            else:
                shape.replace(left=left)
                shape.replace(right=right)
                shape.replace(center=center)

            shape_id = getattr(shape, columns[0])

            return shape_id, shape
                
        def _load(self):
            
            table_name = self.shape_type

            records = SHP_Core._load_records(table_name)
            for record in records:
                (left, right, center) = self._convert_record_to_points(record)
                shape_id, shape = self._convert_record_to_shape(record, left, right, center)
                Shape.set_shape(self.shape_type, shape, shape_id=shape_id)

        # --------------------------------------------------

        def _convert_points_to_tuple(self, left, right, center):
            
            def get_last_id(table_name):

                last_id = 0
                
                SHP_Core._execute("select * from {0} order by id desc limit 1".format(table_name))
                id_pointer = SHP_Core().cursor.fetchone()

                # - 기존 point 가 존재하는 경우                
                if id_pointer != None:
                    # - 마지막 id 추출
                    last_id = id_pointer[0] 

                return last_id

            table_name = self.shape_type + "_PTS"

            left_ids = []
            right_ids = []
            center_ids = []

            for index, points in enumerate([left, right, center]):
                if points != None:
                    points_tuples = [(None, p[0], p[1], p[2]) for p in points]
                    start_id = get_last_id(table_name) + 1

                    SHP_Core._executemany("insert into {0} values(?, ?, ?, ?)".format(table_name), points_tuples)
                    end_id = get_last_id(table_name) 

                    points_ids = list(range(start_id, end_id + 1))
                    [left_ids, right_ids, center_ids][index] += points_ids

            return (left_ids, right_ids, center_ids)

        def _convert_shape_field(self, field, column):
            
            column = column.lower()

            if column == "key":
                converted_field = str(field)
            else:
                converted_field = str(field)

            return converted_field

        def _convert_shape_to_tuple(self, shape, left_ids, right_ids, center_ids):
            
            fields = []

            columns = Shape.get_columns(self.shape_type)            
            # - points column 제외 (마지막 3개 = [left, right, center])
            for index in range(len(columns) - 3):
                column = columns[index]
                field = getattr(shape, column)
                converted_field = self._convert_shape_field(field, column)
                fields.append(converted_field)
            else:
                for points_ids in [left_ids, right_ids, center_ids]:
                    converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), points_ids) if len(points_ids) > 0 else ""
                    fields.append(converted_field)

            shape_tuple = tuple(fields)

            return shape_tuple

        def _save(self):
            
            def delete_table(table_name):
                SHP_Core._delete_all(table_name)
                SHP_Core._delete_all(table_name + "_PTS")

            def insert_tuples(table_name, shape_tuples):
                columns = Shape.get_columns(self.shape_type)
                column_list = reduce(lambda x, y : "{0}, {1}".format(x, y), ["?" for _ in range(len(columns))])
                SHP_Core._executemany("insert into {0} values({1})".format(table_name, column_list), shape_tuples)

            table_name = self.shape_type

            delete_table(table_name)

            shape_tuples = []
            shape_datas = Shape.get_shape_datas(self.shape_type)
            
            for shape_id, shape in shape_datas.items():
                if shape_id not in [-1]:
                    (left, right, center) = (shape.left, shape.right, shape.center)
                    (left_ids, right_ids, center_ids) = self._convert_points_to_tuple(left, right, center)
                    shape_tuple = self._convert_shape_to_tuple(shape, left_ids, right_ids, center_ids)
                    shape_tuples.append(shape_tuple)    

            insert_tuples(table_name, shape_tuples)

    class Crosswalk_Bound(Base):
        
        def _create_shape_table(self, table_name):
            """
            SHP 좌표 테이블 생성
            """
            column_list = "id integer not null primary key unique, x real, y real, z real"

            SHP_Core._create_table(table_name, column_list)

        def _init_table(self):

            def init_record(table_name):

                columns = Shape.get_columns(table_name)

                if not SHP_Core._check_table(table_name):
                    self._create_record_table(table_name, columns)
                    return

                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_record_table(table_name, columns)

            def init_shape(table_name):
                
                table_name = table_name + "_PTS"

                if not SHP_Core._check_table(table_name):
                    self._create_shape_table(table_name)
                    return

                columns = ["x", "y", "z"]
                if not SHP_Core._check_columns(table_name, columns):
                    SHP_Core._drop(table_name)
                    self._create_shape_table(table_name)

            table_name = self.shape_type

            init_record(table_name)
            init_shape(table_name)

        # --------------------------------------------------

        def _convert_record_to_points(self, record):
            
            left = []
            right = []

            for index in [-2, -1]:
                points_field = record[index]
                points_indices = [int(_index) for _index in points_field.split()]
                table_name = self.shape_type + "_PTS"
                start_index = points_indices[0]
                end_index = points_indices[-1]

                SHP_Core._execute("select * from {0} where id >= {1} and id <= {2}".format(table_name, start_index, end_index))
                points_records = SHP_Core().cursor.fetchall()

                for point_row in points_records:
                    point = tuple([float(v) for v in point_row[1:]])
                    side = [left, right][index]
                    side.append(point)

            return (left, right)

        def _convert_record_field(self, field, column):

            column = column.lower()

            if field == None:
                if column == "key":
                    converted_field = None
                else:
                    converted_field = None
            else:
                if column == "key":
                    converted_field = str(field)
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_record_to_shape(self, record, left, right):
            
            shape = Shape.Instance()
            columns = Shape.get_columns(self.shape_type)

            # - points column 을 제외 (마지막 3개 [left, right, center] 제외)
            for index in range(len(columns) - 2):
                column = columns[index]
                field = record[index]
                converted_field = self._convert_record_field(field, column)
                shape.replace(**{column : converted_field})
            else:
                shape.replace(left=left)
                shape.replace(right=right)

            shape_id = getattr(shape, columns[0])

            return shape_id, shape
                
        def _load(self):
            
            table_name = self.shape_type

            records = SHP_Core._load_records(table_name)
            for record in records:
                (left, right) = self._convert_record_to_points(record)
                shape_id, shape = self._convert_record_to_shape(record, left, right)
                Shape.set_shape(self.shape_type, shape, shape_id=shape_id)

        # --------------------------------------------------

        def _convert_points_to_tuple(self, left, right):
            
            def get_last_id(table_name):

                last_id = 0
                
                SHP_Core._execute("select * from {0} order by id desc limit 1".format(table_name))
                id_pointer = SHP_Core().cursor.fetchone()

                # - 기존 point 가 존재하는 경우                
                if id_pointer != None:
                    # - 마지막 id 추출
                    last_id = id_pointer[0] 

                return last_id

            table_name = self.shape_type + "_PTS"

            left_ids = []
            right_ids = []

            for index, points in enumerate([left, right]):
                points_tuples = [(None, p[0], p[1], p[2]) for p in points]
                start_id = get_last_id(table_name) + 1

                SHP_Core._executemany("insert into {0} values(?, ?, ?, ?)".format(table_name), points_tuples)
                end_id = get_last_id(table_name) 

                points_ids = list(range(start_id, end_id + 1))
                [left_ids, right_ids][index] += points_ids
            
            return (left_ids, right_ids)

        def _convert_shape_field(self, field, column):
            
            column = column.lower()

            if column == "key":
                converted_field = str(field)
            else:
                converted_field = str(field)

            return converted_field

        def _convert_shape_to_tuple(self, shape, left_ids, right_ids):
            
            fields = []

            columns = Shape.get_columns(self.shape_type)            
            # - points column 제외 (마지막 3개 = [left, right, center])
            for index in range(len(columns) - 2):
                column = columns[index]
                field = getattr(shape, column)
                converted_field = self._convert_shape_field(field, column)
                fields.append(converted_field)
            else:
                for points_ids in [left_ids, right_ids]:
                    converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), points_ids) if len(points_ids) > 0 else ""
                    fields.append(converted_field)

            shape_tuple = tuple(fields)

            return shape_tuple

        def _save(self):
            
            def delete_table(table_name):
                SHP_Core._delete_all(table_name)
                SHP_Core._delete_all(table_name + "_PTS")

            def insert_tuples(table_name, shape_tuples):
                columns = Shape.get_columns(self.shape_type)
                column_list = reduce(lambda x, y : "{0}, {1}".format(x, y), ["?" for _ in range(len(columns))])
                SHP_Core._executemany("insert into {0} values({1})".format(table_name, column_list), shape_tuples)

            table_name = self.shape_type

            delete_table(table_name)

            shape_tuples = []
            shape_datas = Shape.get_shape_datas(self.shape_type)
            
            for shape_id, shape in shape_datas.items():
                if shape_id not in [-1]:
                    (left, right) = (shape.left, shape.right)
                    (left_ids, right_ids) = self._convert_points_to_tuple(left, right)
                    shape_tuple = self._convert_shape_to_tuple(shape, left_ids, right_ids)
                    shape_tuples.append(shape_tuple)    

            insert_tuples(table_name, shape_tuples)

    class StopLine(Custom):

        def _convert_record_field(self, field, column):
            """
            - link_id
            - lane_id
            """    
        
            column = column.lower()

            if field == None:
                if column == "link_id":
                    converted_field = None
                elif column == "link_id":
                    converted_field = None
                else:
                    converted_field = None
            else:
                if column == "link_id":
                    converted_field = str(field)
                elif column == "link_id":
                    converted_field = str(field)
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - link_id
            - lane_id
            """    
        
            if column == "link_id":
                converted_field = str(field)
            elif column == "lane_id":
                converted_field = str(field)
            else:
                converted_field = str(field)

            return converted_field

    class TrafficLight(Custom):

        def _convert_record_field(self, field, column):
            """
            - key
            - refer_keys
            - stopLine_key
            - lanelet_keys
            """    
        
            column = column.lower()

            if field == None:
                if column == "key":
                    converted_field = None
                elif column == "refer_keys":
                    converted_field = []
                elif column == "stopLine_key":
                    converted_field = None
                elif column == "lanelet_keys":
                    converted_field = []
                else:
                    converted_field = None
            else:
                if column == "key":
                    converted_field = str(field)
                elif column == "refer_keys":
                    converted_field = [str(v) for v in field.split()]
                elif column == "stopLine_key":
                    converted_field = str(field)
                elif column == "lanelet_keys":
                    converted_field = [str(v) for v in field.split()]
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - key
            - refer_keys
            - stopLine_key
            - lanelet_keys
            """    
        
            if column == "key":
                converted_field = str(field)
            elif column == "refer_keys":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            elif column == "stopLine_key":
                converted_field = str(field)
            elif column == "lanelet_keys":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    class TrafficSign(Custom):

        def _convert_record_field(self, field, column):
            """
            - key
            - refer_keys
            - stopLine_key
            - lanelet_keys
            """    
        
            column = column.lower()

            if field == None:
                if column == "key":
                    converted_field = None
                elif column == "refer_keys":
                    converted_field = []
                elif column == "stopLine_key":
                    converted_field = None
                elif column == "lanelet_keys":
                    converted_field = []
                else:
                    converted_field = None
            else:
                if column == "key":
                    converted_field = str(field)
                elif column == "refer_keys":
                    converted_field = [str(v) for v in field.split()]
                elif column == "stopLine_key":
                    converted_field = str(field)
                elif column == "lanelet_keys":
                    converted_field = [str(v) for v in field.split()]
                else:
                    converted_field = str(field)

            return converted_field

        def _convert_shape_field(self, field, column):
            """
            - key
            - refer_keys
            - stopLine_key
            - lanelet_keys
            """    
        
            if column == "key":
                converted_field = str(field)
            elif column == "refer_keys":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            elif column == "stopLine_key":
                converted_field = str(field)
            elif column == "lanelet_keys":
                converted_field = reduce(lambda x, y : "{0} {1}".format(x, y), field) if len(field) > 0 else ""
            else:
                converted_field = str(field)

            return converted_field

    # --------------------------------------------------

    @classmethod
    def get_instance(cls, shape_type):

        instance = None

        origin_types = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5", "C6"]
        origin_types += [_type + "_POST" for _type in origin_types]
        origin_types += ["light_refer", "light_bulb", "sign_refer"]
        custom_types = ["A2_PARENT", "A2_CHILD", "A2P_B2T", "A2_B2", "A1_A2", "endpoint", "road_bound"]

        if shape_type in origin_types:
            template = cls.SHP
        else:
            template = {
                "A2_PARENT" : cls.A2_PARENT,
                "A2_CHILD" : cls.A2_CHILD,
                "A2_B2" : cls.A2_B2,
                "A2P_B2T" : cls.A2P_B2T,
                "A1_A2" : cls.A1_A2,
                "endpoint" : cls.Endpoint,
                "road_bound" : cls.Road_Bound,
                "crosswalk_bound" : cls.Crosswalk_Bound,
                "stopLine" : cls.StopLine,
                "trafficLight" : cls.TrafficLight,
                "trafficSign" : cls.TrafficSign,
            }.get(shape_type)

        if template == None:
            warning_print("Template not found : {0}".format(shape_type))
        else:
            instance = template(shape_type)

        return instance

    @classmethod
    def _init(cls):
        for shape_type in Shape.get_domain():
            instance = cls.get_instance(shape_type)
            if instance != None:
                instance._init_table()

    @classmethod
    def _load(cls):

        domain = Shape.get_domain()
        counter = Process_Counter(len(domain))

        for shape_type in domain:
            instance = cls.get_instance(shape_type)
            if instance != None:
                instance._load()
                counter.add()
            else:
                counter.add(item="warn")
            counter.print_sequence("[{0}] Load shape data".format(shape_type))
        counter.print_result("[{0}] Load shape data".format(shape_type))

    @classmethod
    def _save(cls):

        domain = Shape.get_domain()
        counter = Process_Counter(len(domain))

        for shape_type in domain:
            instance = cls.get_instance(shape_type)
            if instance != None:
                instance._save()
                counter.add()
            else:
                counter.add(item="warn")
            counter.print_sequence("[{0}] Save shape data".format(shape_type))
        counter.print_result("[{0}] Save shape data".format(shape_type))


class SHP_Loader(Loader):

    def _load_data(self, file_path):
        SHP_Core._open(file_path)
        SHP_Core._init()
        SHP_Core._load()
        SHP_Core._close()


class SHP_Saver(Saver):

    def _save_data(self, file_path):
        SHP_Core._open(file_path)
        SHP_Core._init()
        SHP_Core._save()
        SHP_Core._close()
        