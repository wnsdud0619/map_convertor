#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
from collections import (
    defaultdict,
)
from functools import (
    reduce,
)
from copy import deepcopy

import lanelet2
from lanelet2.core import (
    getId, 
    Point3d, 
    LineString3d, 
    Lanelet, 
    AttributeMap, 
    LaneletMap, 
)
from lanelet2.projection import (
    UtmProjector,
)

from scripts.core import core_data 
from scripts.converter import interface
from scripts.functions.coordinate_functions import (
    select_straight,
    calc_length,
    calc_distance,
    create_quad_tree,
    calc_curve_diff,
    check_is_left,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
)

Singleton = core_data.Singleton

# --------------------------------------------------

class Shape(Singleton):

    class Instance():
        
        def replace(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            return self

    column_data = {
        "A1" : ["ID", "AdminCode", "NodeType", "points"],
        "A2" : ["ID", "AdminCode", "RoadRank", "RoadType", "LinkType", "MaxSpeed", "LaneNo", "R_LinkID", "L_LinkID", "FromNodeID", "ToNodeID", "Length", "points"],
        "A3" : ["ID", "AdminCode", "Kind", "RoadType", "points"],
        "A4" : ["ID", "AdminCode", "SubType", "Name", "Direction", "GasStation", "LpgStation", "EvCharger", "Toilet", "points"],
        "A5" : ["ID", "AdminCode", "Type", "SectionID", "points"],
        "B1" : ["ID", "AdminCode", "Type", "SubType", "LinkID", "Ref_Lane", "PostID", "points"],
        "B2" : ["ID", "AdminCode", "Type", "Kind", "R_LinkID", "L_LinkID", "points"],
        "B3" : ["ID", "AdminCode", "Type", "Kind", "LinkID", "points"],
        "C1" : ["ID", "AdminCode", "Type", "LinkID", "Ref_Lane", "PostID", "points"],
        "C3" : ["ID", "AdminCode", "Type", "IsCentral", "LowHigh", "points"],
        "C4" : ["ID", "AdminCode", "Type", "LinkID", "Ref_Lane", "points"],
        "C5" : ["ID", "AdminCode", "Type", "LinkID", "Ref_Lane", "points"],
        "C6" : ["ID", "AdminCode", "Type", "points"],
        # --------------------------------------------------
        "A2_PARENT"                 : ["child_id", "parent_id"],
        "A2_CHILD"                  : ["parent_id", "child_ids"],
        "A2P_B2T"                   : ["link_id", "left_type", "right_type"],
        # --------------------------------------------------
        "endpoint"                  : ["node_id", "left_point", "right_point"],
        "road_bound"                : ["key", "left", "right", "center"],
        # --------------------------------------------------
        "crosswalk_bound"           : ["key", "left", "right"],
        # --------------------------------------------------
        "stopLine"                  : ["link_id", "lane_id"],
        # --------------------------------------------------
        "light_refer"               : ["light_id", "points"],
        "light_bulb"                : ["light_id", "points"],
        "trafficLight"              : ["key", "refer_keys", "stopLine_key", "lanelet_keys"],
        # --------------------------------------------------
        "sign_refer"                : ["sign_id", "points"],
        "trafficSign"               : ["key", "code", "refer_keys", "stopLine_key", "lanelet_keys"],
        # --------------------------------------------------
    }

    def _init_module(self):
        self.data_pack = defaultdict(lambda : dict())

    # --------------------------------------------------

    @classmethod
    def get_domain(cls):

        shape_types = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C3", "C4", "C5", "C6"]

        domain = cls.column_data.keys()
        domain += [shape_type + "_POST" for shape_type in shape_types]

        return domain 

    @classmethod
    def get_columns(cls, shape_type):
        
        if shape_type.split("_")[-1] == "POST":
            return cls.column_data.get(shape_type.split("_")[0])

        return cls.column_data.get(shape_type)

    @classmethod
    def generate_key(cls, shape_type, is_post=True):

        name = "key_datas({0}:{1})".format("post" if is_post else "shape", shape_type)
        
        if not hasattr(cls(), name):
            getter = getattr(cls, "get_{0}_datas".format("post" if is_post else "shape"))
            key_datas = {x : True for x in getter(shape_type).keys()}
            setattr(cls(), name, key_datas)
            
        key_datas = getattr(cls(), name)

        while True:
            keys = key_datas.keys()
            sorted_keys = sorted(keys, key=lambda x : x)
            key = sorted_keys[-1]
            new_key = key[:-5] + str(int(key[-5:]) + 1)
            if key_datas.get(new_key) == None:
                key_datas[new_key] = True
                break

        return new_key

    @classmethod
    def convert_field(cls, column, field):
        """
        SHP 파일 / DB 에 기록된 데이터를 형식에 맞게 변환
        - 1) shape_loader 에서 사용
        - 2) shape_file(Load) 에서 사용
        """

        column = column.lower()

        str_columns = ["id"]
        int_columns = ["type", "rank", "ref", "kind", "code", "laneno", "lowhigh"]
        float_columns = ["length", "maxspeed"]
        bool_columns = ["is"]

        if field == None:
            if any([(item in column) for item in str_columns]):
                converted_field = -1
            elif any([(item in column) for item in int_columns]):
                converted_field = 0
            elif any([(item in column) for item in float_columns]):
                converted_field = 0.0
            elif any([(item in column) for item in bool_columns]):
                converted_field = 0
            else:
                converted_field = None
        else:
            if any([(item in column) for item in str_columns]):
                converted_field = str(field)
            elif any([(item in column) for item in int_columns]):
                converted_field = int(field)
            elif any([(item in column) for item in float_columns]):
                converted_field = float(field)
            elif any([(item in column) for item in bool_columns]):
                converted_field = int(field)
            else:
                if field == "-1":
                    converted_field = -1
                else:
                    converted_field = str(field)

        return converted_field
        
    # --------------------------------------------------

    @classmethod
    def get_shape_datas(cls, shape_type):
        shape_datas = cls().data_pack[shape_type]
        return shape_datas

    @classmethod
    def set_shape_datas(cls, shape_type, shape_datas):
        cls().data_pack[shape_type] = shape_datas

    @classmethod
    def get_shape(cls, shape_type, shape_id):
        shape_datas = cls.get_shape_datas(shape_type)
        if type(shape_datas) == defaultdict:
            shape = shape_datas[shape_id]
            if not hasattr(shape, cls.get_columns(shape_type)[0]):
                shape.replace(**{cls.get_columns(shape_type)[0] : shape_id})
        else:
            shape = shape_datas.get(shape_id)
        return shape

    @classmethod
    def set_shape(cls, shape_type, shape, shape_id=None):
        id = shape.ID if shape_id == None else shape_id
        cls.get_shape_datas(shape_type)[id] = shape

    # --------------------------------------------------

    @classmethod
    def get_post_datas(cls, shape_type):
        post_type = shape_type + "_POST"

        if cls().data_pack.get(shape_type) == None:
            cls.set_shape_datas(post_type, dict())

        return cls().data_pack.get(post_type)

    @classmethod
    def set_post_datas(cls, shape_type, post_datas):
        post_type = shape_type + "_POST"
        cls().data_pack[post_type] = post_datas

    @classmethod
    def get_post(cls, shape_type, shape_id):
        post_datas = cls.get_post_datas(shape_type)
        post = post_datas.get(shape_id) if type(post_datas) != defaultdict else post_datas[shape_id]
        return post

    @classmethod
    def set_post(cls, shape_type, shape, shape_id=None):
        id = shape.ID if shape_id == None else shape_id
        cls.get_post_datas(shape_type)[id] = shape

    # --------------------------------------------------

    @classmethod
    def create_post(cls):

        domain = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C3", "C4", "C5", "C6"]
        counter = Process_Counter(len(domain))
        
        for shape_type in domain:
            
            shape_datas = cls.get_shape_datas(shape_type)

            if not cls.check_shape_source(shape_type):
                counter.add(item="fix")
            else:
                post_datas = deepcopy(shape_datas)
                cls.set_post_datas(shape_type, post_datas)

                if shape_type == "A2":
                    # - Shape_Reviser 에 의해 조정된 A2 길이 갱신
                    for shape in post_datas.values():
                        shape.replace(Length=calc_length(shape.points))

            counter.add()
            counter.print_sequence("Create post ({0})".format(shape_type))

        counter.print_result("Create post")

    @classmethod
    def del_shape(cls, shape_type, shape_id):
        if cls.check_shape(shape_type, shape_id):
            Shape.get_shape_datas(shape_type).pop(shape_id)        

    @classmethod
    def del_post(cls, shape_type, shape_id):
        if cls.check_post(shape_type, shape_id):
            Shape.get_post_datas(shape_type).pop(shape_id)

    # --------------------------------------------------

    @classmethod
    def check_shape(cls, shape_type, shape_id):
        return cls.get_shape_datas(shape_type).get(shape_id) != None

    @classmethod
    def check_post(cls, shape_type, shape_id):
        return cls.get_post_datas(shape_type).get(shape_id) != None

    @classmethod
    def check_shape_source(cls, shape_type):

        shape_datas = cls().data_pack.get(shape_type)

        if shape_datas == None:
            return False
        # elif len(shape_datas) < 1:
        #     return False

        return True

    @classmethod
    def check_post_source(cls, shape_type):
        if cls().data_pack.get(shape_type + "_POST") != None:
            return True
        return False

    # --------------------------------------------------

    @classmethod
    def add_interface(cls):
        """
        Loader ~ Generator 절차 종료 시 수행
        - Converter 데이터 설정 과정
        """
        
        core_data.Altitude = Shape_Interface.Altitude
        core_data.Road = Shape_Interface.Road
        core_data.Crosswalk = Shape_Interface.Crosswalk
        core_data.Light = Shape_Interface.Light
        core_data.Sign = Shape_Interface.Sign


class A1(Singleton):

    @classmethod
    def get_row_nodes(cls, node_id):
        
        (from_links, to_links) = cls.get_row_links(node_id)

        from_ids = [link.FromNodeID for link in from_links]
        to_ids = [link.ToNodeID for link in to_links]

        from_nodes = [Shape.get_post("A1", _id) for _id in from_ids]
        to_nodes = [Shape.get_post("A1", _id) for _id in to_ids]
        
        return (from_nodes, to_nodes)

    @classmethod
    def get_row_links(cls, node_id, is_post=True):

        def get_shape(node_id):

            from_links = []
            to_links = []

            shape = Table.get_instance("A1", "A2", node_id)
            if shape != None:
                from_links = [Shape.get_shape("A2", link_id) for link_id in shape.from_ids]
                to_links = [Shape.get_shape("A2", link_id) for link_id in shape.to_ids]

            return (from_links, to_links)

        def get_post(node_id):

            from_links = []
            to_links = []

            shape = Table.get_instance("A1P", "A2P", node_id)
            if shape != None:
                from_links = [Shape.get_post("A2", link_id) for link_id in shape.from_ids]
                to_links = [Shape.get_post("A2", link_id) for link_id in shape.to_ids]

            return (from_links, to_links)

        return get_shape(node_id) if not is_post else get_post(node_id)

    @classmethod
    def get_side(cls, node_id):

        def get_closest(node, side_nodes):
            
            if len(side_nodes) < 1:
                return None

            min_dist = float("inf")
            min_node = None

            for side_node in side_nodes:
                dist = calc_distance(node.points[0], side_node.points[0])
                if dist < min_dist:
                    min_dist = dist
                    min_node = side_node

            return min_node

        (from_links, to_links) = cls.get_row_links(node_id)

        left_nodes = []
        right_nodes = []

        record = defaultdict(lambda : False)
        for row_index, row_links in enumerate([from_links, to_links]):
            # - From / To 목록
            for row_link in row_links:
                (left_link, right_link) = A2.get_side(row_link)
                # - Left / Right 
                for side_index, side_link in enumerate([left_link, right_link]):
                    if side_link != None:
                        side_nodes = left_nodes if side_index == 0 else right_nodes
                        if not record[side_link.ID]:
                            side_id = side_link.ToNodeID if row_index == 0 else side_link.FromNodeID
                            if side_id not in side_nodes:
                                if side_id != node_id:
                                    side_node = Shape.get_post("A1", side_id)
                                    side_nodes.append(side_node)
                                    record[side_link.ID] = True

        node = Shape.get_post("A1", node_id)

        left_node = get_closest(node, left_nodes)
        right_node = get_closest(node, right_nodes)

        return (left_node, right_node)


class A2(Singleton):

    @classmethod
    def check_intersection(cls, link_id, is_post=True):
        """
        교차로 내 주행로 여부 반환
        """

        checker = Shape.check_shape if not is_post else Shape.check_post
        getter = Shape.get_shape if not is_post else Shape.get_post

        if checker("A2", link_id):
            link = getter("A2", link_id)
            if link.LinkType == 1:
                return True

        return False

    # --------------------------------------------------

    @classmethod
    def get_neighbors(cls, link_id, is_post=True):

        neighbors = []

        getter = Shape.get_post if is_post else Shape.get_shape


        base_link = getter("A2", link_id)
        queue = [base_link]

        while len(queue) > 0:
            
            curr = queue.pop(0)

            for side in [getter("A2", getattr(curr, item)) for item in ["L_LinkID", "R_LinkID"]]:    
                if side != None:
                    if side not in neighbors:
                        queue.append(side)

            neighbors.append(curr)

        return neighbors

    @classmethod
    def get_row_links(cls, link_id, is_post=True):
        
        def get_shape(link_id):

            from_links = []
            to_links = []

            child_ids = cls.get_child_ids(link_id)

            if len(child_ids) < 1:
                child_links = [Shape.get_shape("A2", link_id)]
            else:
                child_links = [Shape.get_post("A2", x) for x in child_ids]

            for child_link in child_links:
                (_from_links, _to_links) = A1.get_row_links(child_link.ToNodeID)
                for _from_link in _from_links:
                    if _from_link.ID not in child_ids:
                        from_links = _from_links
                for _to_link in _to_links:
                    if _to_link.ID not in child_ids:
                        to_links = _to_links

            from_ids = []
            for from_link in from_links:
                origin_id = A2.get_origin_id(from_link.ID)
                if origin_id != None:
                    if origin_id not in from_ids:
                        from_ids.append(origin_id)

            to_ids = []
            for to_link in to_links:
                origin_id = A2.get_origin_id(to_link.ID)
                if origin_id != None:
                    if origin_id not in to_ids:
                        to_ids.append(origin_id)

            from_links = [Shape.get_shape("A2", x) for x in from_ids]
            to_links = [Shape.get_shape("A2", x) for x in to_ids]

            return (from_links, to_links)

        def get_shape(link_id):

            shape = Shape.get_shape("A2", link_id)
            
            from_node_id = shape.FromNodeID
            from_links = A1.get_row_links(from_node_id, is_post=False)[0]

            to_node_id = shape.ToNodeID
            to_links = A1.get_row_links(to_node_id, is_post=False)[1]
            
            return (from_links, to_links)

        def get_post(link_id):

            shape = Shape.get_post("A2", link_id)
            
            from_node_id = shape.FromNodeID
            from_links = A1.get_row_links(from_node_id)[0]

            to_node_id = shape.ToNodeID
            to_links = A1.get_row_links(to_node_id)[1]
            
            return (from_links, to_links)

        return get_post(link_id) if is_post else get_shape(link_id)

    @classmethod
    def select_straight(cls, first, second):
        # - 동일한 직선 정도의 A2 의 경우 입력순서에 따라 값이 달라질 수 있으므로 ID 정렬 수행
        (first, second) = sorted([first, second], key=lambda x : x.ID)
        (main, sub) = (first, second) if select_straight([first.points, second.points], is_index=True) == 0 else (second, first)
        return (main, sub)

    @classmethod
    def get_main(cls, link_id, is_post=True):
        """
        A2 가 속한 이웃목록 내에서 기준 A2 반환
        """

        neighbors = cls.get_neighbors(link_id, is_post=is_post)
        
        main = neighbors[0]
        for neighbor in neighbors[1:]:
            if neighbor.LaneNo < main.LaneNo:
                main = neighbor

        return main 

    @classmethod
    def select_min(cls, links):
        """
        A2 목록 내에서 가장 짧은 A2 반환
        """

        min_length = float("inf")
        min_link = None

        for link in links:
            if link.Length < min_length:
                min_length = link.Length
                min_link = link

        return min_link

    @classmethod
    def classify_pair(cls, first, second):
        (main, sub) = (first, second) if first.LaneNo < second.LaneNo else (second, first)
        return (main, sub)

    # --------------------------------------------------

    @classmethod
    def record_parse(cls, parent_id, child_id):
        
        # 1. 자식->부모 관계 기록
        # - 부모 검색용
        shape = Shape.get_shape("A2_PARENT", child_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(child_id=child_id)
            )
            Shape.set_shape("A2_PARENT", shape, shape_id=child_id)
        shape.replace(parent_id=parent_id)

        # 2. 부모->자식 관계 기록
        # - 자식 검색용
        shape = Shape.get_shape("A2_CHILD", parent_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(parent_id=parent_id)
                .replace(child_ids=[])
            )
            Shape.set_shape("A2_CHILD", shape, shape_id=parent_id)
        shape.child_ids.append(child_id)

    @classmethod
    def get_parent_id(cls, child_id):
        
        parent_id = None

        shape = Shape.get_shape("A2_PARENT", child_id)
        if shape != None:
            parent_id = shape.parent_id

        return parent_id            

    @classmethod
    def get_child_ids(cls, parent_id):

        def get_shape(link_id):
            
            shape = Shape.get_shape("A2_CHILD", link_id)
            if shape == None:
                shape = (Shape.Instance()
                    .replace(parent_id=link_id)
                    .replace(child_ids=[])
                )
                Shape.set_shape("A2_CHILD", shape, shape_id=link_id)
            
            return shape

        child_ids = []

        queue = [parent_id]

        while len(queue) > 0:
            
            link_id = queue.pop(0)

            _child_ids = get_shape(link_id).child_ids

            if len(_child_ids) > 0:
                queue += _child_ids
            else:
                child_ids.append(link_id)
        
        return child_ids

    @classmethod
    def get_origin_id(cls, link_id):

        while cls.get_parent_id(link_id) != None:
            link_id = cls.get_parent_id(link_id)
        return link_id

    # --------------------------------------------------

    @classmethod
    def get_side(cls, link, is_post=True):
        """
        특정 A2 의 좌//우 이웃 A2 반환 (leff_link, right_link)
        """

        getter = Shape.get_post if is_post else Shape.get_shape

        (left_link, right_link) = [getter("A2", getattr(link, item)) for item in ["L_LinkID", "R_LinkID"]]
        return (left_link, right_link)

    # --------------------------------------------------

    @classmethod
    def get_lanes(cls, link_id):

        origin_id = cls.get_origin_id(link_id)

        shape = Table.get_instance("A2", "B2", origin_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(link_id=link_id)
                .replace(left_ids=[])
                .replace(right_ids=[])
            )

        left_lanes = [Shape.get_shape("B2", lane_id) for lane_id in shape.left_ids]
        right_lanes = [Shape.get_shape("B2", lane_id) for lane_id in shape.right_ids]

        return (left_lanes, right_lanes)

    @classmethod
    def get_lane_type(cls, link_id):

        shape = Shape.get_shape("A2P_B2T", link_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(link_id=link_id)
                .replace(left_type=999)
                .replace(right_type=999)
            )
            Shape.set_shape("A2P_B2T", shape, shape_id=link_id)
            
        lane_type = [shape.left_type, shape.right_type]

        return lane_type

    @classmethod
    def set_lane_type(cls, link_id, side_type, side_index):

        shape = Shape.get_shape("A2P_B2T", link_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(link_id=link_id)
                .replace(left_type=999)
                .replace(right_type=999)
            )
            Shape.set_shape("A2P_B2T", shape, shape_id=link_id)
            
        shape.replace(**{["left_type", "right_type"][side_index] : side_type})

    # --------------------------------------------------

    @classmethod
    def check_sub(cls, link_id, side_index, is_post=True):

        link = Shape.get_post("A2", link_id) if is_post else Shape.get_shape("A2", link_id)

        side_link = A2.get_side(link, is_post=is_post)[side_index]
        if side_link != None:
            if link.LaneNo > side_link.LaneNo:
                return True

        return False
                    
    @classmethod
    def check_merge(cls, link_id, side_index, is_post=True):

        link = Shape.get_post("A2", link_id) if is_post else Shape.get_shape("A2", link_id)

        # - 현 방향(side_index)에 대해 Sub 에 해당하는 경우
        if cls.check_sub(link_id, side_index, is_post=is_post):
            side_link = A2.get_side(link, is_post=is_post)[side_index]
            if link.ToNodeID == side_link.ToNodeID:
                return True

        return False

    @classmethod
    def check_branch(cls, link_id, side_index, is_post=True):

        link = Shape.get_post("A2", link_id) if is_post else Shape.get_shape("A2", link_id)

        # - 현 방향(side_index)에 대해 Sub 에 해당하는 경우
        if cls.check_sub(link_id, side_index, is_post=is_post):
            side_link = A2.get_side(link, is_post=is_post)[side_index]
            if link.FromNodeID == side_link.FromNodeID:
                return True

        return False


class A3(Singleton):

    section_datas = {
        "school" : 30.0,
    }

    @classmethod
    def check_schoolzone(cls, section_id):
        instance = Shape.get_shape("A3", section_id)
        if instance != None:
            if instance.Kind == 7:
                return True
        return False

    @classmethod
    def get_speed(cls, section_type):
        speed = cls.section_datas.get(section_type)
        return speed


class B2(Singleton):

    @classmethod
    def get_side_links(cls, lane_id, is_post=True):
        """
        B2 의 좌/우 A2 반환
        """

        left_link = None
        right_link = None

        checker = Shape.check_post if is_post else Shape.check_shape
        getter = Shape.get_post if is_post else Shape.get_shape

        if checker("B2", lane_id):
            lane = getter("B2", lane_id)
            left_link = getter("A2", lane.L_LinkID)
            right_link = getter("A2", lane.R_LinkID)
        
        return (left_link, right_link)

# --------------------------------------------------

class Table(Singleton):

    def create_A1_A2(self):

        node_table = defaultdict(lambda : ([], []))

        node_datas = Shape.get_shape_datas("A1")
        link_datas = Shape.get_shape_datas("A2")

        counter = Process_Counter(len(node_datas))

        for link_id, link in link_datas.items():
            node_table[link.FromNodeID][1].append(link_id)
            node_table[link.ToNodeID][0].append(link_id)
            counter.add(item="fix")
            counter.print_sequence("[Table] Create A1 - A2 table")

        for node_id, (from_ids, to_ids) in node_table.items():
            shape = (Shape.Instance()
                .replace(node_id=node_id)
                .replace(from_ids=from_ids)
                .replace(to_ids=to_ids)
            )
            Shape.set_shape("A1_A2", shape, shape_id=node_id)
            counter.add()
            counter.print_sequence("[Table] Create A1 - A2 table")
        counter.print_result("[Table] Create A1 - A2 table")

    def create_A1P_A2P(self):

        node_table = defaultdict(lambda : ([], []))

        node_datas = Shape.get_post_datas("A1")
        link_datas = Shape.get_post_datas("A2")

        counter = Process_Counter(len(node_datas))

        for link_id, link in link_datas.items():
            node_table[link.FromNodeID][1].append(link_id)
            node_table[link.ToNodeID][0].append(link_id)
            counter.add(item="fix")
            counter.print_sequence("[Table] Create A1 - A2 table (POST)")

        for node_id, (from_ids, to_ids) in node_table.items():
            shape = (Shape.Instance()
                .replace(node_id=node_id)
                .replace(from_ids=from_ids)
                .replace(to_ids=to_ids)
            )
            Shape.set_shape("A1P_A2P", shape, shape_id=node_id)
            counter.add()
            counter.print_sequence("[Table] Create A1 - A2 table (POST)")
        counter.print_result("[Table] Create A1 - A2 table (POST)")

    def create_A2_B2(self):

        lane_table = defaultdict(lambda : ([], []))

        link_datas = Shape.get_shape_datas("A2")
        lane_datas = Shape.get_shape_datas("B2")
        counter = Process_Counter(len(link_datas))

        for lane in lane_datas.values():
            # - 정지선 제외
            if lane.Kind not in [530]:
                for index, item in enumerate(["L_LinkID", "R_LinkID"]):
                    link_id = getattr(lane, item)
                    if Shape.check_shape("A2", link_id):
                        lane_table[link_id][1-index].append(lane.ID) 

        for link_id in link_datas.keys():
            (left_ids, right_ids) = lane_table[link_id]
            if len(left_ids) + len(right_ids) > 0:
                shape = (Shape.Instance()
                    .replace(link_id=link_id)
                    .replace(left_ids=left_ids)
                    .replace(right_ids=right_ids)
                )
                Shape.set_shape("A2_B2", shape, shape_id=link_id)
            counter.add()
            counter.print_sequence("[Table] Create A2 - B2 table")
        counter.print_result("[Table] Create A2 - B2 table")

    def create_A2_C1(self):

        link_datas = Shape.get_shape_datas("A2")
        light_datas = Shape.get_shape_datas("C1")

        counter = Process_Counter(len(link_datas))

        table = defaultdict(lambda : [])
        for light_id, light in light_datas.items():
            link_id = light.LinkID
            if Shape.check_shape("A2", link_id):
                table[link_id].append(light_id)

        for link_id in link_datas.keys():
            light_ids = table[link_id]
            if len(light_ids) > 0:
                instance = (Shape.Instance()
                    .replace(link_id=link_id)
                    .replace(light_ids=light_ids)
                )
                Shape.set_shape("A2_C1", instance, shape_id=link_id)
                counter.add(item="fix")
            counter.add()
            counter.print_sequence("[Table] Create A2 - C1 table")
        counter.print_result("[Table] Create A2 - C1 table")

    def create_C6_C1(self):

        post_datas = Shape.get_shape_datas("C6")
        light_datas = Shape.get_shape_datas("C1")

        counter = Process_Counter(len(post_datas))

        table = defaultdict(lambda : [])
        for light_id, light in light_datas.items():
            post_id = light.PostID
            if Shape.check_shape("C6", post_id):
                table[post_id].append(light_id)

        for post_id in post_datas.keys():
            light_ids = table[post_id]
            if len(light_ids) > 0:
                instance = (Shape.Instance()
                    .replace(post_id=post_id)
                    .replace(light_ids=light_ids)
                )
                Shape.set_shape("C6_C1", instance, shape_id=post_id)
                counter.add(item="fix")
            counter.add()
            counter.print_sequence("[Table] Create C6 - C1 table")
        counter.print_result("[Table] Create C6 - C1 table")

    def create_A2_B1(self):

        link_datas = Shape.get_shape_datas("A2")
        sign_datas = Shape.get_shape_datas("B1")

        counter = Process_Counter(len(link_datas))

        table = defaultdict(lambda : [])
        for sign_id, sign in sign_datas.items():
            link_id = sign.LinkID
            if Shape.check_shape("A2", link_id):
                table[link_id].append(sign_id)

        for link_id in link_datas.keys():
            sign_ids = table[link_id]
            if len(sign_ids) > 0:
                instance = (Shape.Instance()
                    .replace(link_id=link_id)
                    .replace(sign_ids=sign_ids)
                )
                Shape.set_shape("A2_B1", instance, shape_id=link_id)
                counter.add(item="fix")
            counter.add()
            counter.print_sequence("[Table] Create A2 - B1 table")
        counter.print_result("[Table] Create A2 - B1 table")

    def create_C6_B1(self):

        post_datas = Shape.get_shape_datas("C6")
        sign_datas = Shape.get_shape_datas("B1")

        counter = Process_Counter(len(post_datas))

        table = defaultdict(lambda : [])
        for sign_id, sign in sign_datas.items():
            post_id = sign.PostID
            if Shape.check_shape("C6", post_id):
                table[post_id].append(sign_id)

        for post_id in post_datas.keys():
            sign_ids = table[post_id]
            if len(sign_ids) > 0:
                instance = (Shape.Instance()
                    .replace(post_id=post_id)
                    .replace(sign_ids=sign_ids)
                )
                Shape.set_shape("C6_B1", instance, shape_id=post_id)
                counter.add(item="fix")
            counter.add()
            counter.print_sequence("[Table] Create C6 - B1 table")
        counter.print_result("[Table] Create C6 - B1 table")

    def create_A2P_DIR(self):
        """
        A2 의 교차로 방향지시 테이블 생성 
        """

        def get_direction(link):
            """
            교차로 A2 의 방향 반환
            - 1) 회전교차로에 해당 시 [진입 : 좌 / 진출 : 우]
            - 2) 이외 [좌 / 우 / 직진]
            """

            def calc_direction(points, straight_range=30.0):
                """
                특정 좌표목록의 방향성 판별
                - straight / left / right
                - 1) 좌표목록의 최초 선분 각도에서 차이가 30 도 이내라면 straight 로 판단
                - 2) 차이가 -30 이하 = left
                - 3) 차이가 30 이상 = right
                """

                # 1. 시작/종료 각도 차이 추출
                # - 시작각도 = 시작좌표 ~ 시작 + 1 좌표 각도
                # - 종료각도 = 종료 - 1 ~ 종료좌표 각도
                diff = round(calc_curve_diff(points[:2], compare_points=points[-2:]), 5)

                # - 각도차가 30 미만
                if diff < straight_range:
                    direction = "straight"
                # - 각도차가 30 이상
                else:
                    # 2. 좌/우 구분
                    # - 각도차가 180 에 해당하는 경우 "right"
                    direction = "left" if check_is_left(points[0], points[1], points[-1]) else "right"

                return direction

            direction = None

            from_node = Shape.get_post("A1", link.FromNodeID)
            to_node = Shape.get_post("A1", link.ToNodeID)

            # 1) 회전교차로에 해당하는 경우
            if any([x.NodeType == 10 for x in [from_node, to_node]]):
                # - 진입 : 좌회전 표시
                if from_node.NodeType != 10 and to_node.NodeType == 10:
                    direction = "left"
                # - 진출 : 우회전 표시
                elif from_node.NodeType == 10 and to_node.NodeType != 10:
                    direction = "right"

            # 2) 이외
            if direction == None:
                direction = calc_direction(link.points)

            return direction

        link_datas = Shape.get_post_datas("A2")
        counter = Process_Counter(len(link_datas))

        Shape.set_shape_datas("A2P_DIR", dict())
        record = defaultdict(lambda : False)

        for link_id, link in link_datas.items():
            if not record[link_id]:
                # 1) 교차로에 해당하는 경우
                if A2.check_intersection(link_id):
                    # 2) From Node 를 공유하는 A2 목록 개수가 2개 이상
                    dir_links = A1.get_row_links(link.FromNodeID)[1]
                    if len(dir_links) >= 1:
                        for dir_link in dir_links:
                            direction = get_direction(dir_link)
                            instance = (Shape.Instance()
                                .replace(link_id=dir_link.ID)
                                .replace(direction=direction)
                            )
                            Shape.set_shape("A2P_DIR", instance, shape_id=instance.link_id)
                            record[dir_link.ID] = True
                            counter.add(item="fix")
            counter.add()
            counter.print_sequence("[Table] Create A2P - Dir table")
        counter.print_result("[Table] Create A2P - Dir table")

    def create_C1_A2(self):
        """
        C1 : A2 관계 테이블 생성 (1:N)
        """

        def extract_intersection(link_id):
            """
            C1 에 등록된 A2 가 교차로인 경우 추출방식
            - 등록된 A2 는 동일한 정지선을 공유하는 차선 중 가장 왼쪽(LaneNo == 1)에 위치한 것으로 가정
            """
            
            link_ids = []

            # - C1 에 등록된 A2 가 존재하는 경우
            if Shape.check_shape("A2", link_id):
                from_links = A2.get_row_links(link_id, is_post=False)[0]
                if len(from_links) > 0:
                    # 1) From 이웃목록 추출
                    neighbors = A2.get_neighbors(from_links[0].ID, is_post=False)
                    # 2) From 이웃목록의 To 목록 추출
                    # - From 의 To == 동일한 정지선을 고유하는 교차로 차선들
                    for neighbor in neighbors:
                        to_links = A2.get_row_links(neighbor.ID, is_post=False)[1]
                        for to_link in to_links:
                            link_ids.append(to_link.ID)
                    # 3) 중복 제거
                    link_ids = list(set(link_ids))

            return link_ids

        def extract_base(link_id):
            """
            C1 에 등록된 A2 가 교차로가 아닌 경우 추출 방식
            - 등록된 A2 는 동일한 정지선을 공유하는 차선 중 가장 왼쪽(LaneNo == 1)에 위치한 것으로 가정
            """

            link_ids = []

            # - C1 에 등록된 A2 가 존재하는 경우
            if Shape.check_shape("A2", link_id):
                # 1) 이웃목록 추출
                neighbors = A2.get_neighbors(link_id, is_post=False)
                # 2) To 목록 추출
                for neighbor in neighbors:
                    to_links = A2.get_row_links(neighbor.ID, is_post=False)[1]
                    # - To 목록 중 교차로에 해당하는 A2 가 존재 시
                    if any([A2.check_intersection(x.ID, is_post=False) for x in to_links]):
                        # 2-1) 교차로에 해당하는 To A2 를 추출
                        for to_link in to_links:
                            if A2.check_intersection(to_link.ID, is_post=False):
                                link_ids.append(to_link.ID)
                    # - To 목록 중 교차로에 해당하는 A2 가 없는 경우
                    else:
                        # 2-2) 현재 A2 만 추출
                        link_ids.append(neighbor.ID)
                # 3) 중복 제거
                link_ids = list(set(link_ids))

            return link_ids

        light_datas = Shape.get_shape_datas("C1")
        counter = Process_Counter(len(light_datas))

        record = defaultdict(lambda : False)

        for light_id, light in light_datas.items():
            if not record[light_id]:
                link_id = light.LinkID
                if A2.check_intersection(link_id, is_post=False):
                    link_ids = extract_intersection(link_id)
                else:
                    link_ids = extract_base(link_id)
                lights = Light.get_lights(link_id)
                # - 추출된 A2 가 있는 경우 등록
                if len(link_ids) > 0:
                    for light in lights:
                        instance = (Shape.Instance()
                            .replace(light_id=light.ID)
                            .replace(link_ids=link_ids)
                        )
                        Shape.set_shape("C1_A2", instance, shape_id=light.ID)
                        counter.add(item="fix")
                record.update({x.ID : True for x in lights})
            counter.add()
            counter.print_sequence("[Table] Create C1 - A2 table")
        counter.print_result("[Table] Create C1 - A2 table")

    # --------------------------------------------------

    @classmethod
    def get_table(cls, key_type, value_type):
        """
        key 타입 - value 타입에 해당하는 table 반환
        - 예) : A1 : A2
        """

        if not Shape.check_shape_source("{0}_{1}".format(key_type, value_type)):
            getattr(cls(), "create_{0}_{1}".format(key_type, value_type))()

        table = Shape.get_shape_datas("{0}_{1}".format(key_type, value_type))

        return table

    @classmethod
    def get_instance(cls, key_type, value_type, key):
        table = cls.get_table(key_type, value_type)
        instance = table.get(key)
        return instance    
    

class Quad(Singleton):

    def create_A2(self):
        
        counter = Process_Counter(2)

        quad_points = []
        for link in Shape.get_shape_datas("A2").values():
            quad_points += link.points

        quad_table = defaultdict(lambda : [])
        for link in Shape.get_shape_datas("A2").values():
            for point in link.points:
                point2d = (point[0], point[1])
                quad_table[point2d].append(link.ID)
        counter.add()
        counter.print_sequence("[Quad] Create A2 quad")

        quad_tree = create_quad_tree(quad_points)
        counter.add()
        counter.print_result("[Quad] Create A2 quad")
        
        Shape.set_shape_datas("QUAD_A2", (quad_tree, quad_table))

    def create_A2_end(self):

        counter = Process_Counter(2)

        quad_points = []
        for link in Shape.get_shape_datas("A2").values():
            point = link.points[-1]
            point = tuple([round(v, 5) for v in point])
            quad_points.append(point)

        quad_table = defaultdict(lambda : [])
        for link in Shape.get_shape_datas("A2").values():
            point = link.points[-1]
            point = tuple([round(v, 5) for v in point])
            point2d = (point[0], point[1])
            quad_table[point2d].append(link.ID)
        counter.add()
        counter.print_sequence("[Quad] Create A2_end quad")

        quad_tree = create_quad_tree(quad_points)
        counter.add()
        counter.print_result("[Quad] Create A2_end quad")
        
        Shape.set_shape_datas("QUAD_A2_end", (quad_tree, quad_table))

    def create_A2_start(self):

        counter = Process_Counter(2)

        quad_points = []
        for link in Shape.get_shape_datas("A2").values():
            point = link.points[0]
            point = tuple([round(v, 5) for v in point])
            quad_points.append(point)

        quad_table = defaultdict(lambda : [])
        for link in Shape.get_shape_datas("A2").values():
            point = link.points[0]
            point = tuple([round(v, 5) for v in point])
            point2d = (point[0], point[1])
            quad_table[point2d].append(link.ID)
        counter.add()
        counter.print_sequence("[Quad] Create A2_start quad")

        quad_tree = create_quad_tree(quad_points)
        counter.add()
        counter.print_result("[Quad] Create A2_start quad")
        
        Shape.set_shape_datas("QUAD_A2_start", (quad_tree, quad_table))

    def create_A2_edge(self):

        counter = Process_Counter(2)

        quad_points = []
        for link in Shape.get_shape_datas("A2").values():
            for row_index in [0, -1]:
                point = link.points[row_index]
                point = tuple([round(v, 5) for v in point])
                quad_points.append(point)

        quad_table = defaultdict(lambda : [])
        for link in Shape.get_shape_datas("A2").values():
            for row_index in [0, -1]:
                point = link.points[row_index]
                point = tuple([round(v, 5) for v in point])
                point2d = (point[0], point[1])
                quad_table[point2d].append(link.ID)
        counter.add()
        counter.print_sequence("[Quad] Create A2_edge quad")

        quad_tree = create_quad_tree(quad_points)
        counter.add()
        counter.print_result("[Quad] Create A2_edge quad")
        
        Shape.set_shape_datas("QUAD_A2_edge", (quad_tree, quad_table))

    # --------------------------------------------------

    @classmethod
    def get_quad(cls, key_type):
        
        if not Shape.check_shape_source("QUAD_{0}".format(key_type)):
            getattr(cls(), "create_{0}".format(key_type))()

        quad_tree, quad_table = Shape.get_shape_datas("QUAD_{0}".format(key_type))

        return (quad_tree, quad_table)

# --------------------------------------------------

class Endpoint(Singleton):

    def _init_module(self):
        pass

    # --------------------------------------------------

    @classmethod
    def get_endpoint(cls, node_id):

        shape = Shape.get_shape("endpoint", node_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(node_id=node_id)
                .replace(left_point=None)
                .replace(right_point=None)
            )
            Shape.set_shape("endpoint", shape, shape_id=node_id)

        endpoint = [shape.left_point, shape.right_point]

        return endpoint

    @classmethod
    def set_endpoint(cls, node_id, endpoint):
        
        shape = (Shape.Instance()
            .replace(node_id=node_id)
            .replace(left_point=endpoint[0])
            .replace(right_point=endpoint[1])
        )
        Shape.set_shape("endpoint", shape, shape_id=node_id)

    @classmethod
    def check_endpoint(cls, node_id):
        if None in cls.get_endpoint(node_id):
            return False
        return True


class Bound(Singleton):

    @classmethod
    def get_bound(cls, link_id):

        shape = Shape.get_shape("road_bound", link_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(key=link_id)
                .replace(left=None)
                .replace(right=None)
                .replace(center=None)
            )
        left = shape.left
        right = shape.right
        center = shape.center

        bound = [left, right, center]

        return bound

    @classmethod
    def set_bound(cls, link_id, bound):

        shape = (Shape.Instance()
            .replace(key=link_id)
            .replace(left=bound[0])
            .replace(right=bound[1])
            .replace(center=bound[2])
        )
        Shape.set_shape("road_bound", shape, shape_id=link_id)

    @classmethod
    def check_undone(cls, link_id):
        bound = cls.get_bound(link_id)
        if None in bound:
            return True
        return False


class Crosswalk(Singleton):

    @classmethod
    def check_crosswalk(cls, mark_id):
        
        if Shape.get_shape("B3", mark_id):
            if Shape.get_shape("B3", mark_id).Type == 5:
                return True

        return False

    # --------------------------------------------------

    @classmethod
    def get_bound(cls, mark_id):
        
        shape = Shape.get_shape("crosswalk_bound", mark_id)
        if shape == None:
            shape = (Shape.Instance()
                .replace(key=mark_id)
                .replace(left=None)
                .replace(right=None)
            )
        left = shape.left
        right = shape.right

        bound = [left, right]

        return bound

    @classmethod
    def set_bound(cls, mark_id, bound):
        
        shape = (Shape.Instance()
            .replace(key=mark_id)
            .replace(left=bound[0])
            .replace(right=bound[1])
        )
        Shape.set_shape("crosswalk_bound", shape, shape_id=mark_id)


class StopLine(Singleton):

    @classmethod
    def get_key(cls, link_id):

        stopLine_key = None

        shape = Shape.get_shape("stopLine", link_id)

        if shape != None:
            stopLine_key = shape.lane_id

        return stopLine_key

    @classmethod
    def set_key(cls, link_id, stopLine_key):
        shape = (Shape.Instance()
            .replace(link_id=link_id)
            .replace(lane_id=stopLine_key)
        )
        Shape.set_shape("stopLine", shape, link_id)

    @classmethod
    def get_stopLine(cls, link_id=None, stopLine_key=None):

        stopLine = None

        if link_id != None:
            stopLine_key = cls.get_key(link_id)

        if stopLine_key != None:
            shape = Shape.get_shape("B2", stopLine_key)
            if shape != None:
                stopLine = shape.points

        return stopLine    


class Light(Singleton):

    @classmethod
    def check_vehicle(cls, light):
        if light.Type in [11, 12, 13]:
            return False
        return True

    # --------------------------------------------------

    @classmethod
    def get_link(cls, light_id):

        link = None

        shape = Shape.get_shape("C1", light_id)
        if shape != None:
            link_id = shape.LinkID
            link = Shape.get_shape("A2", link_id)

        return link
    
    @classmethod
    def get_lights(cls, link_id=None, post_id=None):
                
        lights = []

        if link_id != None:
            instance = Table.get_instance("A2", "C1", link_id)
        elif post_id != None:
            instance = Table.get_instance("C6", "C1", post_id)
        
        if instance != None:
            light_ids = instance.light_ids
            lights = [Shape.get_shape("C1", x) for x in light_ids]

        return lights

    @classmethod
    def get_post_links(cls, post_id):
        
        links = []

        lights = cls.get_lights(post_id=post_id)
        
        for light in lights:
            link = cls.get_link(light.ID)
            if link != None:
                links.append(link)

        return links

    # --------------------------------------------------

    @classmethod
    def get_refer(cls, light_id):
        
        points = None
        
        shape = Shape.get_shape("light_refer", light_id)
        if shape != None:
            points = shape.points
        
        return points

    @classmethod
    def set_refer(cls, light_id, points):
        shape = (Shape.Instance()
            .replace(light_id=light_id)
            .replace(points=points)
        )
        Shape.set_shape("light_refer", shape, shape_id=light_id)
    
    @classmethod
    def get_bulb(cls, light_id):
        
        points = None
        
        shape = Shape.get_shape("light_bulb", light_id)
        if shape != None:
            points = shape.points

        return points

    @classmethod
    def set_bulb(cls, light_id, points):
        shape = (Shape.Instance()
            .replace(light_id=light_id)
            .replace(points=points)
        )
        Shape.set_shape("light_bulb", shape, shape_id=light_id)
    
    # --------------------------------------------------

    @classmethod
    def get_bulb_width(cls, code):
        """
        전구 가로길이
        """

        if code in [14, 15]:
            bulb_width = 0.2
        else:
            bulb_width = 0.3

        return bulb_width

    @classmethod
    def get_bulb_interval(cls, code):
        """
        좌/우 전구 중심점 간 거리
        """

        if code in [14, 15]:
            bulb_interval = 0.26
        else:
            bulb_interval = 0.355

        return bulb_interval

    @classmethod
    def get_bulb_count(cls, code):
        """
        신호등 전구 가로개수 반환
        """
        bulb_count = {
            1 : 3,
            2 : 4,
            3 : 3,
            4 : 3,
            6 : 2,
            8 : 3,
            9 : 2,
        }.get(code)

        if bulb_count == None:
            bulb_count = 1

        return bulb_count

    @classmethod
    def get_bulb_gap(cls):
        """
        전구 배면판 사이 간격
        """
        return 0.14

    @classmethod
    def get_refer_width(cls, code):

        bulb_count = cls.get_bulb_count(code)
        bulb_interval = cls.get_bulb_interval(code)

        refer_width = float(bulb_count) * bulb_interval

        return refer_width

    @classmethod
    def get_refer_height(cls, light_id):

        light = Shape.get_shape("C1", light_id)
        code = light.Type
        bulb_interval = cls.get_bulb_interval(code)

        height = 0.0

        if code in [1, 2, 8, 9, 10, 99]:
            height = bulb_interval * 1
        elif code in [3, 4, 11, 13]:
            height = bulb_interval * 2
        elif code in [5, 6, 12, 14]:
            height = bulb_interval * 3
        elif code in [7, 15]:
            height = bulb_interval * 4

        return height

    @classmethod
    def get_bulb_colors(cls, light_id):
        
        light = Shape.get_shape("C1", light_id)
        code = light.Type

        colors = []

        if code == 1:
            colors = ["red", "yellow", "green"]
        elif code == 2:
            colors = ["red", "yellow", "green", "green"]
        elif code == 3:
            colors = ["green", "red", "yellow", "green"]
        elif code == 4:
            colors = ["red", "yellow", "green", "red", "yellow", "green"]
        elif code == 5:
            colors = ["green", "red", "yellow"]
        elif code == 6:
            colors = ["red", "yellow", "green", "red", "yellow", "green"]
        elif code == 7:
            colors = ["green", "green", "red", "yellow"]
        elif code == 8:
            colors = ["red", "yellow", "green"]
        elif code == 9:
            colors = ["red", "green"]
        elif code == 10:
            colors = ["yellow"]
        elif code == 11:
            colors = ["red", "green"]
        elif code == 12:
            colors = ["green", "yellow", "red"]
        elif code == 13:
            colors = ["green", "red"]
        elif code == 14:
            colors = ["green", "yellow", "red"]
        elif code == 15:
            colors = ["green", "green", "red", "yellow"]

        return colors

    @classmethod    
    def get_bulb_arrows(cls, light_id):

        light = Shape.get_shape("C1", light_id)
        code = light.Type

        arrows = []

        if code == 1:
            arrows = [None, None, None]
        elif code == 2:
            arrows = [None, None, "left", None]
        elif code == 3:
            arrows = ["left", None, None, None] 
        elif code == 4:
            arrows = ["right", "right", "right", "left", "left", "left"]
        elif code == 5:
            arrows = [None, None, None]
        elif code == 6:
            arrows = ["left", "left", "left", "right", "right", "right"]
        elif code == 7:
            arrows = [None, "left", None, None]
        elif code == 8:
            arrows = [None, None, None]
        elif code == 9:
            arrows = [None, None]
        elif code == 10:
            arrows = [None]
        elif code == 11:
            arrows = [None, None]
        elif code == 12:
            arrows = [None, None, None]
        elif code == 13:
            arrows = [None, None]
        elif code == 14:
            arrows = [None, None, None]
        elif code == 15:
            arrows = [None, "left", None, None]

        return arrows

    # --------------------------------------------------

    @classmethod
    def get_trafficLight(cls, key):

        refer_keys = []
        stopLine_key = None
        lanelet_keys = []

        shape = Shape.get_shape("trafficLight", key)
        if shape != None:
            refer_keys = shape.refer_keys
            stopLine_key = shape.stopLine_key
            lanelet_keys = shape.lanelet_keys

        return (refer_keys, stopLine_key, lanelet_keys)

    @classmethod
    def set_trafficLight(cls, key, refer_keys, stopLine_key, lanelet_keys):

        shape = (Shape.Instance()
            .replace(key=key)
            .replace(refer_keys=refer_keys)
            .replace(stopLine_key=stopLine_key)
            .replace(lanelet_keys=lanelet_keys)
            )
        Shape.set_shape("trafficLight", shape, shape_id=key)


class Sign(Singleton):

    @classmethod
    def get_link(cls, sign_id):

        link = None
        shape = Shape.get_shape("B1", sign_id)
        if shape != None:
            link_id = shape.LinkID
            link = Shape.get_shape("A2", link_id)

        return link

    @classmethod
    def get_signs(cls, link_id=None, post_id=None):

        signs = []

        if link_id != None:
            instance = Table.get_instance("A2", "B1", link_id)
        elif post_id != None:
            instance = Table.get_instance("C6", "B1", post_id)

        if instance != None:
            sign_ids = instance.sign_ids
            signs = [Shape.get_shape("B1", x) for x in sign_ids]
        
        return signs

    @classmethod
    def get_post_links(cls, post_id):

        links = []

        signs = cls.get_signs(post_id=post_id)
        
        for sign in signs:
            link = cls.get_link(sign.ID)
            if link != None:
                links.append(link)

        return links

    @classmethod
    def get_code(cls, sign_id):
        
        code = None
        
        shape = Shape.get_shape("B1", sign_id)
        if shape != None:
            code = shape.SubType

        return code

    # --------------------------------------------------

    @classmethod
    def get_refer_width(cls, sign_id):
        
        def check_triangle(subtype):
            """
            삼각형 여부 반환
            """
            
            if 101 <= subtype <= 140 and subtype != 131:
                return True
            return False

        def check_highway(roadRank):
            """
            고속도로 여부 반환
            """

            if roadRank == 1:
                return True
            return False

        sign = Shape.get_shape("B1", sign_id)
        subtype = sign.SubType
        roadRank = 0

        link = cls.get_link(sign_id)
        if link != None:    
            roadRank = link.RoadRank 

        is_triangle = check_triangle(subtype)
        is_highway = check_highway(roadRank)

        if is_triangle:
            if is_highway:
                width = 1.2
            else:
                width = 0.9
        else:
            if is_highway:
                width = 0.9
            else:
                width = 0.6

        return width

    # --------------------------------------------------
    
    @classmethod
    def get_refer(cls, sign_id):
        points = None
        shape = Shape.get_shape("sign_refer", sign_id)
        if shape != None:
            points = shape.points
        return points

    @classmethod
    def set_refer(cls, sign_id, points):
        shape = (Shape.Instance()
            .replace(sign_id=sign_id)
            .replace(points=points)
        )
        Shape.set_shape("sign_refer", shape, shape_id=sign_id)

    # --------------------------------------------------

    @classmethod
    def get_trafficSign(cls, key):

        code = None
        refer_keys = []
        stopLine_key = None
        lanelet_keys = []

        shape = Shape.get_shape("trafficSign", key)
        if shape != None:
            code = shape.code
            refer_keys = shape.refer_keys
            stopLine_key = shape.stopLine_key
            lanelet_keys = shape.lanelet_keys

        return (code, refer_keys, stopLine_key, lanelet_keys)

    @classmethod
    def set_trafficSign(cls, key, code, refer_keys, stopLine_key, lanelet_keys):

        shape = (Shape.Instance()
            .replace(key=key)
            .replace(code=code)
            .replace(refer_keys=refer_keys)
            .replace(stopLine_key=stopLine_key)
            .replace(lanelet_keys=lanelet_keys)
            )
        Shape.set_shape("trafficSign", shape, shape_id=key)

# --------------------------------------------------

class Shape_Interface():

    class Altitude(interface.Altitude):

        @classmethod
        def get_map_points(cls, map_type):

            domain = {
                "road" : ["A1", "A2", "A3", "A4", "A5", "B2", "B3", "C2", "C3", "C4", "C5", "C6"],
                "regulatory" : ["B1", "C1"],
            }.get(map_type)

            map_points = {}

            for shape_type in domain:
                for shape in Shape.get_shape_datas(shape_type).values():
                    # - 고도값이 0.0 인 경우 제외
                    points = shape.points

                    for point in points:
                        if point[-1] != 0.0:
                            map_points[point] = True

            map_points = map_points.keys()

            return map_points

    class Road(interface.Road):
        
        @classmethod
        def get_keys(cls):
            keys = Shape.get_post_datas("A2").keys()
            return keys

        @classmethod
        def get_side_key(cls, key, side_index):
            shape = Shape.get_post("A2", key)
            side_key = getattr(shape, ["L_LinkID", "R_LinkID"][side_index])
            return side_key

        @classmethod
        def get_bound(cls, key):
            
            bound = [None, None, None]

            shape = Shape.get_shape("road_bound", key)
            if shape != None:
                bound = [shape.left, shape.right, shape.center]
    
            return bound

        @classmethod
        def get_bound_attributes(cls, key, side_index):
            
            def set_type(attributes):
                """
                도로선 두께 설정
                - 대한민국 규격 10 ~ 15 / 15 ~ 20 : 동일하게 취급
                """

                attributes["type"] = "line_thin"

            def set_subtype(attributes, key, side_index):
                """
                도로선 종류 설정
                """

                side_type = A2.get_lane_type(key)[side_index]

                subtype = {
                    11 : "solid", 
                    12 : "dashed", 
                    13 : "dashed_solid",
                    14 : "solid_dashed",
                    21 : "solid_solid", 
                    22 : "dashed",
                    23 : "dashed_solid", 
                    24 : "solid_dashed",
                    99 : "virtual",
                }.get(side_type % 100)

                if subtype in ["virtual"]:
                    attributes["type"] = "virtual"
                else:
                    attributes["type"] = "line_thin"
                    attributes["subtype"] = subtype

            attributes = AttributeMap()
            set_type(attributes)
            set_subtype(attributes, key, side_index)

            return attributes

        @classmethod
        def get_bound3d(cls, key):

            left = None
            right = None
            center = None

            shape = Shape.get_shape("road_bound3d", key)
            if shape != None:
                left = shape.left
                right = shape.right 
                center = shape.center

            bound = [left, right, center]
    
            return bound

        @classmethod
        def set_bound3d(cls, key, bound3d):
            shape = (Shape.Instance()
                .replace(key=key)
                .replace(left=bound3d[0])
                .replace(right=bound3d[1])
                .replace(center=bound3d[2])
            )
            Shape.set_shape("road_bound3d", shape, shape_id=key)

        @classmethod
        def get_lanelet_attributes(cls, key):
            
            def set_type(attributes):
                attributes["type"] = "lanelet"

            def set_location(link, attributes):
                """
                도시지역 / 그외 구분
                """
                attributes["location"] = "urban"

            def set_region(attributes):
                """
                국가코드 설정
                """
                attributes["region"] = "kr"

            def set_subtype(link, attributes):
                """
                일반도로 / 고속도로 구분
                """
                subtype = "road" if link.RoadRank != 1 else "highway"
                attributes["subtype"] = subtype

            def set_speed_limit(link, attributes):
                """
                최대속도 설정
                """
                max_speed = int(link.MaxSpeed)
                if max_speed > 0:
                    attributes["speed_limit"] = "{0}km/h".format(max_speed)

            def set_turn_direction(link, attributes):
                instance = Table.get_instance("A2P", "DIR", link.ID)
                if instance != None:
                    attributes["turn_direction"] = instance.direction

            def set_participants(link, attributes):
                
                #  버스전용차로
                if link.LinkType == 4:
                    attributes["participant:vehicle:bus"] = "yes"

            link = Shape.get_post("A2", key)

            attributes = AttributeMap()
            set_type(attributes)
            set_location(link, attributes)
            set_region(attributes)
            set_subtype(link, attributes)
            set_speed_limit(link, attributes)
            set_turn_direction(link, attributes)
            set_participants(link, attributes)

            return attributes

        @classmethod
        def get_lanelet(cls, key):
            
            lanelet = None
            
            shape = Shape.get_shape("road_lanelet", key)
            if shape != None:
                lanelet = shape.lanelet
    
            return lanelet

        @classmethod
        def set_lanelet(cls, key, lanelet):
            shape = (Shape.Instance()
                .replace(key=key)
                .replace(lanelet=lanelet)
            )
            Shape.set_shape("road_lanelet", shape, shape_id=key)
        
        # --------------------------------------------------

        @classmethod
        def get_row_lanelets(cls, key):
            """
            특정 lanelet의 전/후 lanelet 목록 반환 
            """
            from_links, to_links = A2.get_row_links(key)

            from_lanelets = [cls.get_lanelet(x.ID) for x in from_links]
            to_lanelets = [cls.get_lanelet(x.ID) for x in to_links] 

            return (from_lanelets, to_lanelets)   

    class Crosswalk(interface.Crosswalk):
        
        @classmethod
        def get_keys(cls):
            keys = Shape.get_shape_datas("crosswalk_bound").keys()
            return keys

        @classmethod
        def get_bound(cls, key):
            shape = Shape.get_shape("crosswalk_bound", key)
            bound = [shape.left, shape.right]
            return bound

        @classmethod
        def get_bound_attributes(cls, key, side_index):
            
            def set_type(attributes):
                attributes["type"] = "virtual"

            attributes = AttributeMap()
            set_type(attributes)

            return attributes

        @classmethod
        def get_lanelet_attributes(cls, key):
            
            def set_type(attributes):
                attributes["type"] = "lanelet"

            def set_subtype(attributes):
                attributes["subtype"] = "crosswalk"

            def set_region(attributes):
                attributes["region"] = "kr"

            def set_location(attributes):
                attributes["location"] = "urban"

            def set_one_way(mark, attributes):
                attributes["one_way"] = "false"

            mark = Shape.get_shape("B3", key)

            attributes = AttributeMap()
            set_type(attributes)
            set_subtype(attributes)
            set_region(attributes)
            set_location(attributes)
            set_one_way(mark, attributes)

            return attributes    
        
        @classmethod
        def get_lanelet(cls, key):
            
            lanelet = None
            
            shape = Shape.get_shape("crosswalk_lanelet", key)
            if shape != None:
                lanelet = shape.lanelet
    
            return lanelet

        @classmethod
        def set_lanelet(cls, key, lanelet):
            shape = (Shape.Instance()
                .replace(key=key)
                .replace(lanelet=lanelet)
            )
            Shape.set_shape("crosswalk_lanelet", shape, shape_id=key)
        
    class Light(interface.Light):
    
        @classmethod
        def get_keys(cls):
            keys = Shape.get_shape_datas("trafficLight").keys()
            return keys

        @classmethod
        def get_refer_keys(cls, key):
            
            refer_keys = []

            shape = Shape.get_shape("trafficLight", key)
            if shape != None:
                refer_keys = shape.refer_keys

            return refer_keys            

        @classmethod
        def get_refer(cls, refer_key):
            refer = Light.get_refer(refer_key)
            return refer

        @classmethod
        def get_refer_offset(cls, refer_key):
                
            light = Shape.get_shape("C1", refer_key)
            code = light.Type
            bulb_interval = Light.get_bulb_interval(code)

            offset = 0.0

            if code in [1, 2, 8, 9, 10, 99]:
                offset = -bulb_interval * 0.5
            elif code in [3, 4, 11, 13]:
                offset = -bulb_interval * 1.0
            elif code in [5, 6, 12, 14]:
                offset = -bulb_interval * 1.5
            elif code in [7, 15]:
                offset = -bulb_interval * 2.0

            return offset

        @classmethod
        def get_refer_attributes(cls, refer_key):
            
            def set_type(attributes):
                attributes["type"] = "traffic_light"

            def set_height(attributes, refer_key):
                attributes["height"] = str(Light.get_refer_height(refer_key))

            attributes = AttributeMap()
            set_type(attributes)
            set_height(attributes, refer_key)
            
            return attributes

        @classmethod
        def get_bulb(cls, refer_key):
            bulb = Light.get_bulb(refer_key)
            return bulb

        @classmethod
        def get_bulb_offset(cls, refer_key, index):
            
            light = Shape.get_shape("C1", refer_key)
            code = light.Type
            bulb_interval = Light.get_bulb_interval(code)

            offset = 0.0

            if code in [1, 2, 8, 9, 10, 99]:
                offset = 0.0
            elif code in [3]:
                offset = [bulb_interval * -0.5, 0, 0, 0][index]
            elif code in [4]:
                offset = [
                    bulb_interval * -0.5,
                    bulb_interval * -0.5,
                    bulb_interval * -0.5,
                    bulb_interval * 0.5,
                    bulb_interval * 0.5,
                    bulb_interval * 0.5,
                    ][index]
            elif code in [5, 12, 14]:
                offset = [
                    bulb_interval * -1.0,
                    bulb_interval * -0.0,
                    bulb_interval * 1.0
                ][index]
            elif code in [6]:
                offset = [
                    bulb_interval * -1.0,
                    bulb_interval * -0.0,
                    bulb_interval * 1.0,
                    bulb_interval * -1.0,
                    bulb_interval * -0.0,
                    bulb_interval * 1.0,
                ][index]
            elif code in [7, 15]:
                offset = [
                    bulb_interval * -1.5,
                    bulb_interval * -0.5,
                    bulb_interval * 0.5,
                    bulb_interval * 1.5,
                ][index]
            elif code in [11, 13]:
                offset = [
                    bulb_interval * -0.5,
                    bulb_interval * 0.5
                ][index]

            return offset

        @classmethod
        def get_bulb_attributes(cls, refer_key):
            
            def set_type(attributes):
                attributes["type"] = "light_bulbs"

            def set_subtype(attributes, refer_key):
                colors = Light.get_bulb_colors(refer_key)
                attributes["subtype"] = reduce(lambda x, y : "{0}_{1}".format(x, y), colors) if len(colors) > 0 else ""

            def set_traffic_light_id(attributes, refer_key):
                refer3d = cls.get_refer3d(refer_key)
                attributes["traffic_light_id"] = str(refer3d.id)

            attributes = AttributeMap()
            set_type(attributes)
            set_subtype(attributes, refer_key)
            set_traffic_light_id(attributes, refer_key)

            return attributes

        @classmethod
        def get_bulb_color(cls, refer_key, index):
            color = Light.get_bulb_colors(refer_key)[index]
            return color

        @classmethod
        def get_bulb_arrow(cls, refer_key, index):
            arrow = Light.get_bulb_arrows(refer_key)[index]
            return arrow

        @classmethod
        def get_refer_origin(cls, refer_key):
            light = Shape.get_shape("C1", refer_key)
            origin = light.points[0]
            return origin

        @classmethod
        def get_stopLine_key(cls, key):
            
            stopLine_key = None

            shape = Shape.get_shape("trafficLight", key)
            if shape != None:
                stopLine_key = shape.stopLine_key

            return stopLine_key
    
        @classmethod
        def get_stopLine(cls, stopLine_key):
            stopLine = StopLine.get_stopLine(stopLine_key=stopLine_key)
            return stopLine

        @classmethod
        def get_stopLine_attributes(cls, stopLine_key):
            
            def set_type(attributes):
                attributes["type"] = "stop_line"

            attributes = AttributeMap()
            set_type(attributes)
            
            return attributes

        @classmethod
        def get_lanelet_keys(cls, key):
            
            lanelet_keys = []
            
            shape = Shape.get_shape("trafficLight", key)
            if shape != None:
                lanelet_keys = shape.lanelet_keys

            return lanelet_keys

        @classmethod
        def get_regulatory_attributes(cls, key):

            def set_type(attributes):
                attributes["type"] = "regulatory_element"

            def set_subtype(attributes):
                attributes["subtype"] = "traffic_light"

            attributes = AttributeMap()
            set_type(attributes)
            set_subtype(attributes)

            return attributes

        # --------------------------------------------------

        @classmethod
        def get_refer3d(cls, refer_key):
            
            refer3d = None
            
            shape = Shape.get_shape("trafficLight_refer3d", refer_key)
            if shape != None:
                refer3d = shape.refer3d
            
            return refer3d

        @classmethod
        def set_refer3d(cls, refer_key, refer3d):
            
            shape = (Shape.Instance()
                .replace(refer_key=refer_key)
                .replace(refer3d=refer3d)
            )
            Shape.set_shape("trafficLight_refer3d", shape, shape_id=refer_key)

        @classmethod
        def get_bulb3d(cls, refer_key):
            
            bulb3d = None
            
            shape = Shape.get_shape("trafficLight_bulb3d", refer_key)
            if shape != None:
                bulb3d = shape.bulb3d
            
            return bulb3d

        @classmethod
        def set_bulb3d(cls, refer_key, bulb3d):
            
            shape = (Shape.Instance()
                .replace(refer_key=refer_key)
                .replace(bulb3d=bulb3d)
            )
            Shape.set_shape("trafficLight_bulb3d", shape, shape_id=refer_key)

        @classmethod
        def get_stopLine3d(cls, stopLine_key):

            stopLine3d = None

            shape = Shape.get_shape("stopLine3d", stopLine_key)
            if shape != None:
                stopLine3d = shape.stopLine3d

            return stopLine3d

        @classmethod
        def set_stopLine3d(cls, stopLine_key, stopLine3d):

            shape = (Shape.Instance()
                .replace(key=stopLine_key)
                .replace(stopLine3d=stopLine3d)
            )
            Shape.set_shape("stopLine3d", shape, shape_id=stopLine_key)

        @classmethod
        def get_regulatory(cls, key):
            
            regulatory = None

            shape = Shape.get_shape("trafficLight_regulatory", key)
            if shape != None:
                regulatory = shape.regulatory

            return regulatory

        @classmethod
        def set_regulatory(cls, key, regulatory):
            shape = (Shape.Instance()
                .replace(key=key)
                .replace(regulatory=regulatory)
            )
            Shape.set_shape("trafficLight_regulatory", shape, shape_id=key)
        
    class Sign(interface.Sign):
    
        @classmethod
        def get_keys(cls):
            keys = Shape.get_shape_datas("trafficSign").keys()
            return keys

        @classmethod
        def get_refer_keys(cls, key):
            
            refer_keys = []

            shape = Shape.get_shape("trafficSign", key)
            if shape != None:
                refer_keys = shape.refer_keys

            return refer_keys            

        @classmethod
        def get_refer(cls, refer_key):
            refer = Sign.get_refer(refer_key)
            return refer

        @classmethod
        def get_refer_offset(cls, refer_key):
            offset = 0.0
            return offset

        @classmethod
        def get_refer_attributes(cls, refer_key):
            
            def set_type(attributes):
                attributes["type"] = "traffic_sign"

            attributes = AttributeMap()
            set_type(attributes)

            return attributes

        @classmethod
        def get_refer_origin(cls, refer_key):
            sign = Shape.get_shape("B1", refer_key)
            origin = sign.points[0]
            return origin

        @classmethod
        def get_stopLine_key(cls, key):
            
            stopLine_key = None

            shape = Shape.get_shape("trafficSign", key)
            if shape != None:
                stopLine_key = shape.stopLine_key

            return stopLine_key
    
        @classmethod
        def get_stopLine(cls, stopLine_key):
            stopLine = StopLine.get_stopLine(stopLine_key=stopLine_key)
            return stopLine

        @classmethod
        def get_stopLine_attributes(cls, stopLine_key):
            
            def set_type(attributes):
                attributes["type"] = "stop_line"

            attributes = AttributeMap()
            set_type(attributes)
            
            return attributes

        @classmethod
        def get_lanelet_keys(cls, key):
            
            lanelet_keys = []
            
            shape = Shape.get_shape("trafficSign", key)
            if shape != None:
                lanelet_keys = shape.lanelet_keys

            return lanelet_keys

        @classmethod
        def get_nation_code(cls, key):

            nation_code = None

            shape = Shape.get_shape("trafficSign", key)
            if shape != None:
                code = shape.code
                nation_code = "kr" + str(code)

            return nation_code

        @classmethod
        def get_regulatory_attributes(cls, key):
            
            def set_type(attributes):
                attributes["type"] = "regulatory_element"

            def set_subtype(attributes):
                attributes["subtype"] = "traffic_sign"

            attributes = AttributeMap()
            set_type(attributes)
            set_subtype(attributes)            

            return attributes

        # --------------------------------------------------

        @classmethod
        def get_refer3d(cls, refer_key):
            
            refer3d = None
            
            shape = Shape.get_shape("trafficSign_refer3d", refer_key)
            if shape != None:
                refer3d = shape.refer3d
            
            return refer3d

        @classmethod
        def set_refer3d(cls, refer_key, refer3d):
            
            shape = (Shape.Instance()
                .replace(refer_key=refer_key)
                .replace(refer3d=refer3d)
            )
            Shape.set_shape("trafficSign_refer3d", shape, shape_id=refer_key)

        @classmethod
        def get_stopLine3d(cls, stopLine_key):

            stopLine3d = None

            shape = Shape.get_shape("stopLine3d", stopLine_key)
            if shape != None:
                stopLine3d = shape.stopLine3d

            return stopLine3d

        @classmethod
        def set_stopLine3d(cls, stopLine_key, stopLine3d):

            shape = (Shape.Instance()
                .replace(key=stopLine_key)
                .replace(stopLine3d=stopLine3d)
            )
            Shape.set_shape("stopLine3d", shape, shape_id=stopLine_key)

        @classmethod
        def get_regulatory(cls, key):
            
            regulatory = None

            shape = Shape.get_shape("trafficSign_regulatory", key)
            if shape != None:
                regulatory = shape.regulatory

            return regulatory

        @classmethod
        def set_regulatory(cls, key, regulatory):
            shape = (Shape.Instance()
                .replace(key=key)
                .replace(regulatory=regulatory)
            )
            Shape.set_shape("trafficSign_regulatory", shape, shape_id=key)
        
# --------------------------------------------------

class Convert(Singleton):

    def _init_module(self, map_name):
        self.map_name = map_name

    # --------------------------------------------------

    @classmethod
    def create_map(cls):
        return LaneletMap()

    @classmethod
    def convert_to_point3d(cls, point):
        return Point3d(getId(), point[0], point[1], point[2])

    @classmethod
    def convert_to_lineString3d(cls, points, link_id=None):
        line3d = LineString3d(getId(), [cls.convert_to_point3d(point) for point in points])
        if link_id != None:
            line3d.attributes["ID"] = str(link_id)
        return line3d

    @classmethod
    def convert_to_lanelet(cls, left_bound, right_bound, center_line=None):
        lanelet = Lanelet(
            getId(), 
            cls.convert_to_lineString3d(left_bound),
            cls.convert_to_lineString3d(right_bound)
        )
        if center_line != None:
            lanelet.centerline = cls.convert_to_lineString3d(center_line)
        return lanelet

    @classmethod
    def save_map(cls, module_name, file_name, map, sub_dir=None):

        # 1. main.py dir_path 추출
        base_path = reduce(lambda x, y : "{0}/{1}".format(x, y), os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])

        # 2. dir_path 추출
        dir_path = base_path + "/map/{0}/{1}".format(cls().map_name, module_name)

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

