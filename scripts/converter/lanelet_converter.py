#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import shutil
from collections import defaultdict
from functools import reduce
from scipy.spatial import (
    Delaunay
)
import numpy as np
import math

import lanelet2
from lanelet2.core import (
    getId, 
    Point3d, 
    LineString3d, 
    Lanelet, 
    AttributeMap, 
    TrafficSign,
    TrafficSignsWithType, 
    TrafficLight,
    LaneletMap, 
)
from lanelet2.projection import (
    UtmProjector,
)

from scripts.core import core_data 
from scripts.file.file_data import Core
from scripts.functions.file_functions import (
    open_file,
    check_file,
    create_directory,
)
from scripts.functions.coordinate_functions import (
    calc_z_from_plane,
    check_inside,
    create_quad_tree,
    get_closest_quad_point,
    check_colinear,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
    class_print,
)


Singleton = core_data.Singleton
Altitude = None
Road = None
Crosswalk = None
Light = None
Sign = None


class Triangle(Core):

    def load_triangles(self, map_type):

        def open_file():
            # 1. DB 연결
            file_path = "{0}/triangles/{1}/triangle.db".format(self.base_path, self.source_path)
            self._open(file_path)

        def check_file(map_type):
            """
            - 1) 삼각형 목차 테이블 검사
            - 2) 삼각형 좌표 테이블 검사
            """

            def check_index():
                """
                - 1) 파일 내 테이블 존재유무 검사
                - 2) 파일 내 테이블 항목 검사
                """

                table_name = map_type
                columns = ["index", "points"]

                if not self._check_table(table_name):
                    return False

                if not self._check_columns(table_name, columns):
                    return False

                return True

            def check_points():
                """
                - 1) 파일 내 테이블 존재유무 검사
                - 2) 파일 내 테이블 항목 검사
                """
    
                table_name = map_type + "_PTS"
                columns = ["id", "x", "y", "z"]

                if not self._check_table(table_name):
                    return False

                if not self._check_columns(table_name, columns):
                    return False

                return True

            return check_index() and check_points()

        def read_file(map_type):
            """
            삼각 데이터 추출
            """

            def read_points(record, map_type):
                
                points = []

                table_name = map_type + "_PTS"

                points_field = record[-1]
                points_indices = [int(index) for index in points_field.split()]

                start_id = points_indices[0]
                end_id = points_indices[-1]

                self._execute("select * from {0} where id >= {1} and id <= {2}".format(table_name, start_id, end_id))
                points_records = self.cursor.fetchall()

                for point_row in points_records:
                    point = tuple([float(v) for v in point_row[1:]])
                    points.append(point)

                return points

            def read_index(record):
                
                index = int(record[0])

                return index

            def create_triangle(index, points):

                triangle = {
                    "index" : index,
                    "points" : points
                }

                return triangle

            triangles = []

            table_name = map_type

            # 1. 목차 테이블 레코드 추출
            records = self._load_records(table_name)

            for record in records:
                # - 현 목차 레코드에 해당하는 좌표목록 추출
                points = read_points(record)
                # - 현 목차 레코드에 해당하는 순번 추출
                index = read_index(record)
                # - 삼각 데이터 생성
                tri = create_triangle(index, points)
                # - 삼각 데이터 등록
                triangles.append(tri)

            return triangles

        def close_file():
            self._close()

        triangles = None

        # 1. 연결
        open_file()
        # 2. 검사
        # - DB 파일이 존재하는 경우
        if check_file(map_type):
            # 3. 추출
            triangles = read_file(map_type)
            # 4. 종료
            close_file()

        return triangles

    def save_triangles(self, triangles, map_type):
        
        def open_file():
            # 1. DB 연결
            file_path = "{0}/triangles/{1}/triangle.db".format(self.base_path, self.source_path)
            self._open(file_path)

        def init_file(map_type):

            def init_index(map_type):
                
                def create_table():
                    column_list = "id integer, points text"
                    self._create_table(table_name, column_list)

                table_name = map_type

                # 1. 테이블 없음 : 테이블 생성
                if not self._check_table(table_name):
                    create_table()

                # 2. 테이블 초기화
                self._delete_all(table_name)

            def init_points(map_type):

                def create_table():
                    column_list = "id integer not null primary key unique, x real, y real, z real"
                    self._create_table(table_name, column_list)

                table_name = map_type + "_PTS"

                # 1. 테이블 없음 : 테이블 생성
                if not self._check_table(table_name):
                    create_table()

                # 2. 테이블 초기화
                self._delete_all(table_name)

            # 1. 목차 테이블 초기화
            init_index(map_type)
            # 2. 좌표 테이블 초기화
            init_points(map_type)

        def write_file(map_type, triangles):
            
            def write_points(triangle, map_type):
                """
                좌표 데이터 기록
                """

                def get_last_id(table_name):
                    """
                    좌표 테이블의 현재 마지막 레코드 순번을 추출
                    """

                    last_id = 0
                    
                    self._execute("select * from {0} order by id desc limit 1".format(table_name))
                    id_pointer = self.cursor.fetchone()

                    # - 기존 point 가 존재하는 경우                
                    if id_pointer != None:
                        # - 마지막 id 추출
                        last_id = id_pointer[0] 

                    return last_id

                table_name = map_type + "_PTS"

                # 1. 좌표 tuple 목록 생성
                points_tuples = []
                for point in triangle["points"]:
                    points_tuple = (None, point[0], point[1], point[2])
                    points_tuples.append(points_tuple)

                # 2-1. 좌표 기록 
                # 2-2. 좌표 목차 목록 추출
                start_id = get_last_id(table_name) + 1
                self._executemany("insert into {0} values(?, ?, ?, ?)".format(table_name), points_tuples)
                end_id = get_last_id(table_name)
                points_ids = list(range(start_id, end_id + 1))

                return points_ids

            def write_index(triangle, points_ids):
                """
                목차 데이터 기록
                """

                # 1. 목차 추출
                index = triangle["index"]

                # 2. 목차 tuple 생성
                points = reduce(lambda x, y : "{0} {1}".format(x, y), points_ids)

                # 3. 기록 
                self._execute("insert into {0}(id, points) values({1}, '{2}')".format(table_name, index, points))

            table_name = map_type

            for tri in triangles:
                # - 좌표 데이터 기록
                # - 좌표 목차 추출
                points_ids = write_points(tri, map_type)
                # - 목차 데이터 기록
                write_index(tri, points_ids)

        def close_file():
            self._close()

        # 1. 연결
        open_file()
        # 2. 초기화
        init_file(map_type)
        # 3. 기록
        write_file(map_type, triangles)
        # 4. 종료
        close_file()

    def create_triangles(self, base_path, source_path):
        
        def create(map_type):

            def except_points(points, range_unit=150.0, max_degree=10.0):
                """
                오류값, 극단값에 해당하는 고도(z)를 가진 좌표를 제외한다.
                - 전체 좌표를 구간으로 나눠서 극단값을 제외한다.
                """

                # - 중복 좌표 제거
                points = list(set(points))

                # 1. 구간 너비(range_unit) + 인접 좌표간 최대 각도차(max_degree)를 기반으로 인접 좌표간 최대 고도차(max_diff) 추출 
                max_diff = math.tan(math.radians(max_degree)) * (range_unit / 2.0)

                # 2. 구간 분할 (quad)
                root_node = create_quad_tree(points, unit=range_unit)
                
                # 3. 각 구간별 극단값 제외
                _points = []
                queue = [root_node]
                while len(queue) > 0:
                    # 2-1) 현재 노드 선택
                    node = queue.pop(0)
                    # 2-2) 단말이 아닐 생략
                    if len(node["child_nodes"]) > 0:
                        # - 큐에 자식을 추가
                        queue += node["child_nodes"] 
                    # 2-3) 단말인 경우 극단값 제외
                    elif len(node["points"]) > 0:
                        # - 평균 고도 구하기
                        average_alt = np.mean([x[2] for x in node["points"]])
                        # - 평균 고도와 최대 고도차의 범위를 벗어나는 좌표를 제거목록에 등록
                        _points += [x for x in node["points"] if average_alt - max_diff < x[2] < average_alt + max_diff]

                return _points

            def create_triangles(map_points):
                """
                좌표목록을 들로네 삼각분할을 이용해 삼각화
                """

                triangles = []

                # 1. x, y 좌표만 추출
                xy_points = np.array([(x[0], x[1]) for x in map_points])

                # 2. 삼각화
                _delaunay = Delaunay(np.array(xy_points))
                
                for indices in _delaunay.simplices:
                    
                    points = []
                    for index in indices:
                        point = map_points[index]
                        points.append(point)

                    if not check_colinear(points[0], points[1], points[2]):
                        triangle = {
                            "index" : len(triangles),
                            "points" : points,
                        }
                        triangles.append(triangle)

                return triangles            

            def create_table(triangles):
                """
                좌표 - 삼각형 연관 테이블 생성
                - key : 좌표
                - value : 삼각형 목차 (index) 목록
                """
                # 1. 테이블 초기화
                tri_table = defaultdict(lambda : [])
                # 2. 좌표 - 삼각형 목차 기록
                [tri_table[point].append(triangle["index"]) for triangle in triangles for point in triangle["points"]]

                return tri_table

            if not hasattr(self, "quad_datas"):
                self.quad_datas = dict()
                self.triangle_datas = dict()
                self.table_datas = dict()

            counter = Process_Counter(1)
            counter.print_sequence("[{0}] Create altitude map ({1})".format(self.__class__.__name__, map_type))

            # 1. 좌표목록 추출
            map_points = Altitude.get_map_points(map_type)
            # 2. 좌표목록 필터링 (극단값 제외)
            map_points = except_points(map_points)
            # 3. Quad 트리 생성
            quad_tree = create_quad_tree(map_points)
            # 4. 삼각화 데이터 생성
            triangles = self.load_triangles(map_type)
            # - 기존 데이터가 없는 경우 (Load 실패)
            if triangles == None:
                # - 1) 좌표목록에 기반 삼각화 데이터 생성
                triangles = create_triangles(map_points)
                # - 2) 삼각화 데이터 기록
                self.save_triangles(triangles, map_type)
            # 5. 삼각화 테이블 생성 (좌표 -> 삼각형 추적)            
            tri_table = create_table(triangles)
        
            self.quad_datas[map_type] = quad_tree
            self.triangle_datas[map_type] = triangles
            self.table_datas[map_type] = tri_table

            counter.add()
            counter.print_result("[{0}] Create altitude map ({1})".format(self.__class__.__name__, map_type))

        self.base_path = base_path
        self.source_path = source_path

        map_types = ["road", "regulatory"]

        for map_type in map_types:
            create(map_type)

    # --------------------------------------------------

    @classmethod
    def get_altitude(cls, point, map_type):

        def get_closest(point):
            """
            전체 삼각형 좌표 중 좌표(point)에 최근접 삼각형 좌표 추출
            """

            quad_tree = cls().quad_datas[map_type]
            closest_p = get_closest_quad_point(point, quad_tree)
            return closest_p

        def get_triangles(point):

            _triangles = []

            triangles = cls().triangle_datas[map_type]
            tri_table = cls().table_datas[map_type]

            # 1. 최근접 좌표 추출
            closest_p = get_closest(point)

            # 2. 최근접좌표 포함 삼각형 목록 추출
            tri_indices = tri_table[closest_p]
            for tri_index in tri_indices:
                tri = triangles[tri_index]
                _triangles.append(tri)

            return _triangles

        def get_inner(point, triangles):
            """
            삼각형 목록 중에서 좌표(point)를 포함하는 삼각형 추출
            """

            triangle = None

            for tri in triangles:
                if check_inside(point, tri["points"]):
                    triangle = tri
                    break

            return triangle

        def get_offset():

            offset = {
                "DGIST" : 28.7389845089,
                "Techno_Scenario" : 28.7389845089,
                "Techno_Scenario2" : 28.7389845089,
                "Techno" : 28.7389845089,
            }.get(cls().source_path)

            if offset == None:
                offset = 0.0

            return offset

        altitude = None

        # 1. 좌표 포함 삼각형 목록 추출
        triangles = get_triangles(point)
        
        # 2. 좌표포함 삼각형 추출
        triangle = get_inner(point, triangles)

        # - 포함 삼각형 존재 시
        if triangle != None:
            # - 삼각형 기반 고도 추출
            altitude = calc_z_from_plane(triangle["points"], point[0], point[1])
        
        # - 고도 추출 실패 (삼각형 검색 실패 or 삼각형 계산 오류)
        if altitude == None:
            # - 최근접 좌표 고도 사용
            altitude = get_closest(point)[2]

        # 3. 고도 offset 적용
        altitude = altitude + get_offset()

        return altitude


class Tracker(Singleton):

    def _init_module(self):
        self.create_id_table()

    def create_id_table(self):

        def set_id(source_id, converted_id):
            id_table[source_id] = converted_id

        id_table = dict()

        for source_id in Road.get_keys():
            converted_id = self.convert_id(source_id)
            set_id(source_id, converted_id)

        for source_id in Crosswalk.get_keys():
            converted_id = self.convert_id(source_id)
            set_id(source_id, converted_id)

        for key in Light.get_keys():
            for source_id in Light.get_refer_keys(key):
                converted_id = self.convert_id(source_id)
                set_id(source_id, converted_id)
                
        for key in Sign.get_keys():
            for source_id in Sign.get_refer_keys(key):
                converted_id = self.convert_id(source_id)
                set_id(source_id, converted_id)

        self.id_table = id_table

    # --------------------------------------------------

    @classmethod
    def convert_id(cls, source_id):
    
        def get_unfolded(source_id, serial_length=6):
            """
            분할과정을 거친 id 펼치기 ("_" 제거)
            - 일련번호 길이(serial_length) : SHP = 6 
            """

            # 1. id 펼치기
            parsed_id = source_id.split("_")
            # 2. "_" 를 제거한 id 합치기
            unfolded_id = reduce(lambda x, y : x + y, source_id.split("_"), "")
            # 3. 개체번호 길이 계산
            tail_length = serial_length + len(parsed_id) - 1

            return unfolded_id, tail_length

        def get_compact(unfolded_id, tail_length):

            layer_name = unfolded_id[:2]
            serial_number = unfolded_id[-tail_length:]

            compact_id = layer_name + serial_number

            return compact_id

        def str_to_num(string):
            """
            알파벳 포함 문자열을 숫자로 변환
            - 알파벳, 숫자 이외는 포함되지 않은 것으로 전제
            """
            
            num = 0

            for index, char in enumerate(string[::-1].lower()):
                try:
                    _num = int(char)
                except ValueError:
                    _num = ord(char) - 96
                num += _num * 10**index

            return num

        # 1. 분할과정 거친 id 펼치기 + 개체번호 길이(tail_length) 추출 
        (unfolded_id, tail_length) = get_unfolded(source_id)

        # 2. 개체형식 + 개체번호 id 추출
        compact_id = get_compact(unfolded_id, tail_length)

        # 3. id 문자열 -> 숫자 변환            
        num_id = str_to_num(compact_id)

        return num_id

    @classmethod
    def get_id(cls, source_id=None):
        
        def get_random():
            """
            id_table 에 등록된 적 없는 신규 id 반환
            """

            id = None

            id_table = getattr(cls(), "id_table")

            while True:
                id = getId() 
                if id_table.get(id) == None:
                    id_table[id] = True
                    break

            return id

        def get_converted(source_id):

            id = None

            id_table = getattr(cls(), "id_table")

            # 1. 기존에 등록된 id 추출
            id = id_table.get(source_id) 
            # - source_id 에 대응하는 기존 id 가 없는 경우
            if id == None:
                # 2. source_id 변환 후 반환
                id = cls.convert_id(source_id)
                id_table[source_id] = id

            return id

        id = get_random() if source_id == None else get_converted(source_id)
    
        return id


class Convert(Singleton):

    def _init_module(self, base_path, source_path):

        global Altitude
        global Road
        global Crosswalk
        global Light
        global Sign

        Altitude = core_data.Altitude
        Road = core_data.Road
        Crosswalk = core_data.Crosswalk
        Light = core_data.Light
        Sign = core_data.Sign
        
        self.base_path = base_path
        self.source_path = source_path

        Tracker()
        Triangle().create_triangles(base_path, source_path)

    # --------------------------------------------------

    @classmethod
    def create_map(cls):
        return LaneletMap()

    @classmethod
    def convert_to_point3d(cls, point, map_type="road"):
        altitude = Triangle.get_altitude(point, map_type)
        point3d = Point3d(Tracker.get_id(), point[0], point[1], altitude)
        return point3d

    @classmethod
    def convert_to_lineString3d(cls, points, key=None, attributes=AttributeMap(), convert_type=None):

        def share_point3d(point, convert_type, map_type):
            
            if not hasattr(cls, "point3d:{0}".format(convert_type)):
                setattr(cls, "point3d:{0}".format(convert_type), dict())
            
            point3d_datas = getattr(cls, "point3d:{0}".format(convert_type))
            point2d = (round(point[0], 5), round(point[1], 5))
            point3d = point3d_datas.get(point2d)

            if point3d == None:
                point3d = cls.convert_to_point3d(point, map_type=map_type)
                point3d_datas[point2d] = point3d

            return point3d            

        map_type = {
            "Road" : "road",
            "Crosswalk" : "road",
            "TrafficLight" : "regulatory",
            "TrafficSign" : "regulatory",
            None : "road",
        }.get(convert_type)

        line3d = LineString3d(Tracker.get_id(source_id=key), [], attributes)
        if convert_type in ["Road"]:
            for index, point in enumerate(points):
                if index in [0, len(points) - 1]:
                    point3d = share_point3d(point, convert_type, map_type)
                else:
                    point3d = cls.convert_to_point3d(point, map_type=map_type)
                line3d.append(point3d)
        else:
            for point in points:
                point3d = cls.convert_to_point3d(point, map_type=map_type)
                line3d.append(point3d)

        return line3d

    @classmethod
    def convert_to_lanelet(cls, left, right, key=None, attributes=AttributeMap(), center_line=None, convert_type=None):
        
        left3d = cls.convert_to_lineString3d(left, convert_type=convert_type)
        right3d = cls.convert_to_lineString3d(right, convert_type=convert_type)
        lanelet = Lanelet(Tracker.get_id(source_id=key), left3d, right3d, attributes)

        if center_line != None:
            center3d = cls.convert_to_lineString3d(center_line, convert_type=convert_type)
            lanelet.centerline = center3d

        return lanelet

    @classmethod
    def create_lineString3d(cls, point3ds, key=None, attributes=AttributeMap()):
        line3d = LineString3d(Tracker.get_id(source_id=key), point3ds, attributes)
        return line3d

    @classmethod
    def create_lanelet(cls, left3d, right3d, key=None, attributes=AttributeMap(), center3d=None):
        lanelet = Lanelet(Tracker.get_id(source_id=key), left3d, right3d, attributes)
        if center3d != None:
            lanelet.centerline = center3d
        return lanelet

    @classmethod
    def save_map(cls, file_name, map, sub_dir=None):

        # 1. main.py dir_path 추출
        base_path = reduce(lambda x, y : "{0}/{1}".format(x, y), os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])

        # 2. dir_path 추출
        dir_path = base_path + "/map/{0}/Converter".format(cls().source_path)

        # 2.5 sub_dir 적용
        dir_path = "{0}/{1}".format(dir_path, sub_dir) if sub_dir != None else dir_path

        # 3. directory 검사
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 4. file_path 추출
        file_path = dir_path + "/{0}".format(file_name)

        # 5. Save
        projector = UtmProjector(lanelet2.io.Origin(0.00, 126.0), False, False)
        lanelet2.io.write(file_path, map, projector)

    @classmethod
    def clear_directory(cls, dir_path):
        """
        .../Converter 내부의 디렉토리 제거
        """
        
        path = reduce(lambda x, y : "{0}/{1}".format(x, y), os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
        path = path + "/map/{0}/Converter".format(cls().source_path)
        path = path + "/" + dir_path
        
        try:
            shutil.rmtree(path)
        except OSError as e:
            warning_print("Cannot find such path : {0}".format(path))

    # --------------------------------------------------
    

class Converter(Singleton):

    def _init_module(self, base_path, source_path):
        Convert(base_path, source_path)

    # --------------------------------------------------

    class Road():

        def create_bound3d(self):
        
            def share(key, side_index):
                side_key = Road.get_side_key(key, side_index)
                side3d = Road.get_bound3d(side_key)[1-side_index]

                return side3d

            def create(key, side, side_index):

                attributes = Road.get_bound_attributes(key, side_index)
                side3d = Convert.convert_to_lineString3d(side, attributes=attributes, convert_type="Road")

                return side3d

            lanelet_keys = Road.get_keys()
            counter = Process_Counter(len(lanelet_keys))            

            for key in lanelet_keys:
                [left, right, center] = Road.get_bound(key)
                if None not in [left, right]:
                    bound3d = [None, None, None]
                    for side_index in [0, 1]:
                        side3d = share(key, side_index)
                        if side3d == None:
                            side3d = create(key, [left, right][side_index], side_index)
                        bound3d[side_index] = side3d
                    center3d = None
                    if center != None:
                        center3d = Convert.convert_to_lineString3d(center, convert_type="Road")
                        bound3d[2] = center3d
                    Road.set_bound3d(key, bound3d)
                    counter.add()
                else:
                    counter.add(item="warn")
                counter.print_sequence("[{0}] Convert bound".format(self.__class__.__name__))
            counter.print_result("[{0}] Convert bound".format(self.__class__.__name__))

        def create_lanelet(self):
            
            def create(key, left3d, right3d, center3d):
                attributes = Road.get_lanelet_attributes(key)
                # lanelet = Convert.create_lanelet(left3d, right3d, key=key, attributes=attributes, center3d=center3d)
                lanelet = Convert.create_lanelet(left3d, right3d, key=key, attributes=attributes)
                return lanelet

            lanelet_keys = Road.get_keys()
            counter = Process_Counter(len(lanelet_keys))

            for key in lanelet_keys:
                [left3d, right3d, center3d] = Road.get_bound3d(key) 
                if None not in [left3d, right3d]:
                    lanelet = create(key, left3d, right3d, center3d)
                    Road.set_lanelet(key, lanelet)
                    counter.add()
                else:
                    counter.add(item="warn")
                counter.print_sequence("[{0}] Create lanelet".format(self.__class__.__name__))
            counter.print_result("[{0}] Create lanelet".format(self.__class__.__name__))

        # --------------------------------------------------

        def execute(self):

            def save():
                
                map = Convert.create_map()

                keys = Road.get_keys()
                for key in keys:
                    lanelet = Road.get_lanelet(key)
                    if lanelet != None:
                        map.add(lanelet)

                Convert.save_map("lanelet.osm", map, sub_dir="Road")

            self.create_bound3d()
            self.create_lanelet()
        
            save()

    class Crosswalk():

        def create_lanelet(self):

            def create(key):

                bound3d = []

                bound = Crosswalk.get_bound(key)
                [left, right] = bound

                for side_index, side in enumerate([left, right]):
                    bound_attributes = Crosswalk.get_bound_attributes(key, side_index)
                    side3d = Convert.convert_to_lineString3d(side, attributes=bound_attributes, convert_type="Crosswalk")
                    bound3d.append(side3d)

                lanelet_attributes = Crosswalk.get_lanelet_attributes(key)
                lanelet = Convert.create_lanelet(bound3d[0], bound3d[1], key=key, attributes=lanelet_attributes)
                return lanelet

            keys = Crosswalk.get_keys()
            counter = Process_Counter(len(keys))

            for key in keys:
                lanelet = create(key)
                Crosswalk.set_lanelet(key, lanelet)
                counter.add()
                counter.print_sequence("[{0}] Create lanelet".format(self.__class__.__name__))
            counter.print_result("[{0}] Create lanelet".format(self.__class__.__name__))

        # --------------------------------------------------

        def execute(self):

            def save():
                
                map = Convert.create_map()

                keys = Crosswalk.get_keys()
                for key in keys:
                    lanelet = Crosswalk.get_lanelet(key)
                    map.add(lanelet)

                Convert.save_map("crosswalk_lanelet.osm", map, sub_dir="Crosswalk")

            self.create_lanelet()
        
            save()
    
    class TrafficLight():

        def create_regulatory(self):

            def create(key):

                def get_refer3ds(refer_keys):
                    
                    def set_refer_altitudes(refer_key, refer3d):
                        
                        refer_origin = Light.get_refer_origin(refer_key)
                        refer_origin3d = Convert.convert_to_point3d(refer_origin, map_type="regulatory")
                        altitude = refer_origin3d.z
                        refer_offset = Light.get_refer_offset(refer_key)

                        for point3d in refer3d:
                            point3d.z = altitude + refer_offset

                    refer3ds = []

                    for refer_key in refer_keys:

                        refer3d = Light.get_refer3d(refer_key)
                        if refer3d == None:
                            refer = Light.get_refer(refer_key)
                            refer_attributes = Light.get_refer_attributes(refer_key)
                            refer3d = Convert.convert_to_lineString3d(refer, key=refer_key, attributes=refer_attributes, convert_type="TrafficLight")
                        set_refer_altitudes(refer_key, refer3d)

                        Light.set_refer3d(refer_key, refer3d)
                        refer3ds.append(refer3d)

                    return refer3ds

                def get_bulb3ds(refer_keys):

                    def set_point_attributes(refer_key, bulb3d):
                        
                        for bulb_index, point3d in enumerate(bulb3d):
                            color = Light.get_bulb_color(refer_key, bulb_index)
                            arrow = Light.get_bulb_arrow(refer_key, bulb_index)
                            point3d.attributes["color"] = color
                            if arrow != None:
                                point3d.attributes["arrow"] = arrow

                    def set_bulb_altitudes(refer_key, bulb3d):

                        refer_origin = Light.get_refer_origin(refer_key)
                        refer_origin3d = Convert.convert_to_point3d(refer_origin, map_type="regulatory")
                        altitude = refer_origin3d.z

                        for index, point3d in enumerate(bulb3d):
                            bulb_offset = Light.get_bulb_offset(refer_key, index)
                            point3d.z = altitude + bulb_offset

                    bulb3ds = []

                    for refer_key in refer_keys:

                        bulb3d = Light.get_bulb3d(refer_key)
                        if bulb3d == None:
                            bulb = Light.get_bulb(refer_key)
                            bulb_attributes = Light.get_bulb_attributes(refer_key)
                            bulb3d = Convert.convert_to_lineString3d(bulb, attributes=bulb_attributes, convert_type="TrafficLight")

                        set_point_attributes(refer_key, bulb3d)
                        set_bulb_altitudes(refer_key, bulb3d)                        

                        Light.set_bulb3d(refer_key, bulb3d)
                        bulb3ds.append(bulb3d)

                    return bulb3ds

                def get_stopLine3d(stopLine_key):

                    stopLine3d = Light.get_stopLine3d(stopLine_key)
                    if stopLine3d == None:
                        stopLine = Light.get_stopLine(stopLine_key)
                        if stopLine != None:
                            stopLine_attributes = Light.get_stopLine_attributes(stopLine_key)
                            stopLine3d = Convert.convert_to_lineString3d(stopLine, key=stopLine_key, attributes=stopLine_attributes)
                            Light.set_stopLine3d(stopLine_key, stopLine3d)
                    
                    return stopLine3d

                def get_lanelets(lanelet_keys):
                    
                    lanelets = []

                    for lanelet_key in lanelet_keys:
                        lanelet = Road.get_lanelet(lanelet_key)
                        if lanelet != None:
                            lanelets.append(lanelet)

                    return lanelets

                def set_stopLine3d(regulatory, stopLine3d):
                    if stopLine3d != None:
                        regulatory.stopLine = stopLine3d

                def set_lanelets(regulatory, lanelets):
                    for lanelet in lanelets:
                        lanelet.addRegulatoryElement(regulatory)

                regulatory = None

                refer_keys = Light.get_refer_keys(key)
                stopLine_key = Light.get_stopLine_key(key)
                lanelet_keys = Light.get_lanelet_keys(key)

                refer3ds = get_refer3ds(refer_keys)
                bulb3ds = get_bulb3ds(refer_keys)
                stopLine3d = get_stopLine3d(stopLine_key)
                lanelets = get_lanelets(lanelet_keys)

                regulatory_attributes = Light.get_regulatory_attributes(key)
                regulatory = TrafficLight(Tracker.get_id(), regulatory_attributes, refer3ds + bulb3ds)

                set_stopLine3d(regulatory, stopLine3d)
                set_lanelets(regulatory, lanelets)

                return regulatory

            keys = Light.get_keys()
            counter = Process_Counter(len(keys))

            for key in keys:
                regulatory = create(key)
                Light.set_regulatory(key, regulatory)
                counter.add()
                counter.print_sequence("[{0}] Create regulatory".format(self.__class__.__name__))
            counter.print_result("[{0}] Create regulatory".format(self.__class__.__name__))
            
        # --------------------------------------------------
    
        def execute(self):

            def save_regulatory():

                map = Convert.create_map()
                
                keys = Light.get_keys()
                for key in keys:
                    regulatory = Light.get_regulatory(key)
                    map.add(regulatory)

                Convert.save_map("regulatory.osm", map, sub_dir="Regulatory/TrafficLight")

            self.create_regulatory()

            save_regulatory()

    class TrafficSign():

        def create_regulatory(self):

            def create(key):

                def get_refer3ds(refer_keys):
                    
                    def set_refer_altitudes(refer_key, refer3d):
                        
                        refer_origin = Sign.get_refer_origin(refer_key)
                        refer_origin3d = Convert.convert_to_point3d(refer_origin, map_type="regulatory")
                        altitude = refer_origin3d.z
                        refer_offset = Sign.get_refer_offset(refer_key)

                        for point3d in refer3d:
                            point3d.z = altitude + refer_offset

                    refer3ds = []

                    for refer_key in refer_keys:

                        refer3d = Sign.get_refer3d(refer_key)
                        if refer3d == None:
                            refer = Sign.get_refer(refer_key)
                            refer_attributes = Sign.get_refer_attributes(refer_key)
                            refer3d = Convert.convert_to_lineString3d(refer, key=refer_key, attributes=refer_attributes, convert_type="TrafficSign")
                        set_refer_altitudes(refer_key, refer3d)

                        Sign.set_refer3d(refer_key, refer3d)
                        refer3ds.append(refer3d)

                    return refer3ds

                def get_stopLine3d(stopLine_key):

                    stopLine3d = Sign.get_stopLine3d(stopLine_key)
                    if stopLine3d == None:
                        stopLine = Sign.get_stopLine(stopLine_key)
                        if stopLine != None:
                            stopLine_attributes = Sign.get_stopLine_attributes(stopLine_key)
                            stopLine3d = Convert.convert_to_lineString3d(stopLine, key=stopLine_key, attributes=stopLine_attributes)
                            Sign.set_stopLine3d(stopLine_key, stopLine3d)
                    
                    return stopLine3d

                def get_lanelets(lanelet_keys):
                    
                    lanelets = []

                    for lanelet_key in lanelet_keys:
                        lanelet = Road.get_lanelet(lanelet_key)
                        if lanelet != None:
                            lanelets.append(lanelet)

                    return lanelets

                def set_stopLine3d(regulatory, stopLine3d):
                    if stopLine3d != None:
                        regulatory.addRefLine(stopLine3d)

                def set_lanelets(regulatory, lanelets):
                    for lanelet in lanelets:
                        lanelet.addRegulatoryElement(regulatory)

                regulatory = None

                refer_keys = Sign.get_refer_keys(key)
                stopLine_key = Sign.get_stopLine_key(key)
                lanelet_keys = Sign.get_lanelet_keys(key)

                refer3ds = get_refer3ds(refer_keys)
                stopLine3d = get_stopLine3d(stopLine_key)
                lanelets = get_lanelets(lanelet_keys)

                nation_code = Sign.get_nation_code(key)
                trafficSignsWithType = TrafficSignsWithType(refer3ds, nation_code)

                regulatory_attributes = Sign.get_regulatory_attributes(key)
                regulatory = TrafficSign(Tracker.get_id(), regulatory_attributes, trafficSignsWithType)

                set_stopLine3d(regulatory, stopLine3d)
                set_lanelets(regulatory, lanelets)

                return regulatory                

            keys = Sign.get_keys()
            counter = Process_Counter(len(keys))

            for key in keys:
                regulatory = create(key)
                Sign.set_regulatory(key, regulatory)
                counter.add()
                counter.print_sequence("[{0}] Create regulatory".format(self.__class__.__name__))
            counter.print_result("[{0}] Create regulatory".format(self.__class__.__name__))
            
        # --------------------------------------------------

        def execute(self):
            self.create_regulatory()

    # --------------------------------------------------

    class Saver():

        def save_road(self, map):
            
            keys = Road.get_keys()

            for key in keys:
                lanelet = Road.get_lanelet(key)
                if lanelet != None:
                    map.add(lanelet)

        def save_crosswalk(self, map):

            keys = Crosswalk.get_keys()
            for key in keys:
                lanelet = Crosswalk.get_lanelet(key)
                map.add(lanelet)

        def save_light(self, map):
            
            keys = Light.get_keys()
            for key in keys:
                regulatory = Light.get_regulatory(key)
                map.add(regulatory)

        def save_sign(self, map):
            
            keys = Sign.get_keys()
            for key in keys:
                regulatory = Sign.get_regulatory(key)
                map.add(regulatory)

        # --------------------------------------------------

        def revise_light_bulb(self):

            def get_bulb3d_datas():
                
                bulb3d_datas = {}

                keys = Light.get_keys()
                for key in keys:
                    refer_keys = Light.get_refer_keys(key)
                    for refer_key in refer_keys:
                        bulb3d = Light.get_bulb3d(refer_key)
                        bulb3d_datas[bulb3d.id] = bulb3d.attributes["traffic_light_id"]

                return bulb3d_datas

            def get_path():

                base_path = reduce(lambda x, y : "{0}/{1}".format(x, y), os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
                dir_path = base_path + "/map/{0}/Converter".format(Convert().source_path)
                file_path = dir_path + "/lanelet2_map.osm"

                return file_path

            revise_list = []

            bulb3d_datas = get_bulb3d_datas()
            counter = Process_Counter(len(bulb3d_datas))

            file_path = get_path()
            map_file = open_file(file_path, "r")
            readlines = map_file.readlines()
            map_file.close()

            for index, line in enumerate(readlines):
                if 'role="refers"' in line:
                    bulb3d_id = int(line.split('"')[3])
                    traffic_light_id = bulb3d_datas.get(bulb3d_id)
                    if traffic_light_id != None:
                        revise_data = {
                            "line_index" : index,
                            "line" : '    <member type="way" ref="{0}" role="light_bulbs" />\n'.format(bulb3d_id)
                        }
                        revise_list.append(revise_data)
                        counter.add(item="fix")
                        counter.print_sequence("[Checker] Revise light bulb")

            for revise_data in revise_list:
                line_index = revise_data["line_index"]
                line = revise_data["line"]
                readlines[line_index] = line

                counter.add()
                counter.print_sequence("[Checker] Revise light bulb")

            map_file = open_file(get_path(), "w")
            map_file.writelines(readlines)
            map_file.close()

            counter.print_result("[Checker] Revise light bulb")

        # --------------------------------------------------

        def execute(self):

            map = Convert.create_map()

            self.save_road(map)
            self.save_crosswalk(map)
            self.save_light(map)
            self.save_sign(map)

            Convert.save_map("lanelet2_map.osm", map)

            self.revise_light_bulb()

    # --------------------------------------------------

    class Validator():
        """
        맵 검증 class
        """

        def __init__(self):
            Convert.clear_directory("Validator")

        # --------------------------------------------------

        def validate_Lanelet(self):

            def validate_connection():
                """
                Lanelet 전/후 연결 확인 (Point3d 공유 확인)
                - leftBound
                - rightBound
                - centerline
                """

                keys = Road.get_keys()
                counter = Process_Counter(len(keys))

                # 1. Point3d 공유 테이블 생성
                share_table = defaultdict(lambda : ([], []))
                point_table = dict()

                for key in keys:
                    lanelet = Road.get_lanelet(key)
                    if lanelet != None:
                        for item in ["leftBound", "rightBound", "centerline"]:
                            lineString3d = getattr(lanelet, item)
                            for row_index in [0, -1]:
                                point3d = lineString3d[row_index]
                                share_table[point3d.id][-1-row_index].append(lanelet.id)
                                point_table[point3d.id] = point3d
                    counter.add()
                    counter.print_sequence("[Validator : Lanelet] Validate connection")

                # 2. 전/후 Lanelet 이 둘다 존재하지 않는 Point3d 추출
                disconnections = []
                for id, (from_ids, to_ids) in share_table.items():
                    if len(from_ids) < 1 or len(to_ids) < 1:
                        point3d = point_table.get(id)
                        disconnections.append(point3d)
                        counter.add(item="warn")
                        counter.print_sequence("[Validator : Lanelet] Validate connection")

                # - 추출된 단절이 1개 이상인 경우
                if len(disconnections) > 0:

                    # 3. 기록
                    map = Convert.create_map()
                    for point3d in disconnections:
                        point = (point3d.x, point3d.y, point3d.z)
                        _point3d = Convert.convert_to_point3d(point)
                        map.add(_point3d)

                    Convert.save_map("Lanelet(disconnection).osm", map, sub_dir="Validator/Lanelet")

                counter.print_result("[Validator : Lanelet] Validate connection")

            validate_connection()

        # --------------------------------------------------

        def execute(self):
            self.validate_Lanelet()

    # --------------------------------------------------
        
    @classmethod
    def execute(cls, base_path, source_path):

        class_print("{0}".format(cls.__name__))
        cls(base_path, source_path)

        cls.Road().execute()
        cls.Crosswalk().execute()
        cls.TrafficLight().execute()
        cls.TrafficSign().execute()

        cls.Saver().execute()
        cls.Validator().execute()

