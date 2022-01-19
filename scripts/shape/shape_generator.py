#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from six import add_metaclass
from abc import (
    ABCMeta,
    abstractmethod,
)
from collections import defaultdict
from functools import reduce

from scripts.shape.shape_data import (
    Shape,
    Table,
    Quad,
    A1,   
    A2,
    Endpoint,
    Bound,
    Crosswalk,
    Shape_Interface,
    StopLine,
    Light,
    Sign,
    Convert,
)
from scripts.shape.shape_module import Module
from scripts.functions.coordinate_functions import (
    calc_degree,
    calc_average_degree,
    convert_to_vector,
    correct_misdirected,
    get_closest_quad_point,
    get_intersection_on_points,
    calc_distance,
    calc_distance_from_line,
    get_ortho_line,
    get_mid,
    calc_length,
    improve_points_density,
    move_line,
    check_is_left,
    simplify_polygon,
    check_intersection_on_points,
    get_closest_segment,
    rotate_seg,
    convert_point_to_line,
    move_point,
    check_same,
    sample_points,
    get_closest_point,
    reduce_polygon,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
)


class Road_Process():
    
    def get_vertical_degree(self, node_id):

        if not hasattr(self, "vertical_degrees"):
            setattr(self, "vertical_degrees", dict())

        vertical_degrees = getattr(self, "vertical_degrees")
        vertical_degree = vertical_degrees.get(node_id)

        if vertical_degree == None:

            (from_links, to_links) = A1.get_row_links(node_id)

            from_degrees = []
            for from_link in from_links:
                deg = calc_degree(from_link.points[-2], from_link.points[-1])
                from_degrees.append(deg)

            to_degrees = []
            for to_link in to_links:
                deg = calc_degree(to_link.points[0], to_link.points[1])
                to_degrees.append(deg)

            vertical_degree = calc_average_degree(from_degrees + to_degrees)
            vertical_degrees[node_id] = vertical_degree

        return vertical_degree

    def set_vertical_degree(self, node_id, degree):
        
        if not hasattr(self, "vertical_degrees"):
            setattr(self, "vertical_degrees", dict())

        vertical_degrees = getattr(self, "vertical_degrees")
        vertical_degrees[node_id] = degree

    def get_horizontal(self, node_id, dist=25.0):
        
        point = Shape.get_post("A1", node_id).points[0]                
        vertical_deg = self.get_vertical_degree(node_id)

        horizontal = get_ortho_line(point, degree=vertical_deg, dist1=dist, dist2=dist)

        return horizontal

    def get_vertical(self, node_id):
        
        point = Shape.get_post("A1", node_id).points[0]
        vertical_deg = self.get_vertical_degree(node_id)

        vertical_vec = convert_to_vector(vertical_deg)
        vertical = [(0,0,0), vertical_vec]
        vertical = move_line(vertical, start=vertical[0], end=point)

        return vertical


class Road_Generator():

    class Endpoint_Generator():

        class Generate_Process(Road_Process):

            options = {
                "Generate_From_Lane" : "lane",
                "Generate_From_Neighbor" : "neighbor",
                "Generate_From_Row" : "row",
                "Generate_From_Side" : "side",
            }

            def get_option(self):
                return self.__class__.options.get(self.__class__.__name__)

            # --------------------------------------------------

            def update_neighbor(self, node_id, side_index):
                
                (left_node, right_node) = A1.get_side(node_id)
                side_node = (left_node, right_node)[side_index]

                if side_node != None:
                    endpoint = Endpoint.get_endpoint(node_id)
                    side_endpoint = Endpoint.get_endpoint(side_node.ID)
                    side_endpoint[1-side_index] = endpoint[side_index]
                    Endpoint.set_endpoint(side_node.ID, side_endpoint)

            # --------------------------------------------------

            @abstractmethod
            def _generate_endpoint(self, node_id):
                pass

            # --------------------------------------------------

            def execute(self):
                
                def function_test():

                    def row_link_test():

                        map = Convert.create_map()

                        record = dict()

                        for node_id in node_datas.keys():
                            (from_links, to_links) = A1.get_row_links(node_id)
                            
                            for from_link in from_links:
                                line3d = record.get(from_link.ID)
                                if line3d == None:
                                    line3d = Convert.convert_to_lineString3d(from_link.points)
                                    line3d.attributes["ID"] = str(from_link.ID)
                                    record[from_link.ID] = line3d
                                line3d.attributes["ToNode : {0}".format(node_id)] = ""
                                line3d.attributes["ToNode (Link) : {0}".format(from_link.ToNodeID)] = ""

                                map.add(line3d)

                            for to_link in to_links:
                                line3d = record.get(to_link.ID)
                                if line3d == None:
                                    line3d = Convert.convert_to_lineString3d(to_link.points)
                                    line3d.attributes["ID"] = str(to_link.ID)
                                    record[to_link.ID] = line3d
                                line3d.attributes["FromNode : {0}".format(node_id)] = ""
                                line3d.attributes["FromNode (Link) : {0}".format(to_link.FromNodeID)] = ""

                                map.add(line3d)

                        Convert.save_map("Shape_Generator", "row_link.osm", map)                        

                    def row_node_test():
                        
                        map = Convert.create_map()

                        for node_id, node in node_datas.items():

                            (from_nodes, to_nodes) = A1.get_row_nodes(node_id)
                            from_ids = [x.ID for x in from_nodes]
                            to_ids = [x.ID for x in to_nodes]

                            point3d = Convert.convert_to_point3d(node.points[0])
                            point3d.attributes["ID"] = str(node_id)
                            point3d.attributes["FromNodes : {0}".format(from_ids)] = ""
                            point3d.attributes["ToNodes : {0}".format(to_ids)] = ""

                            map.add(point3d)

                        Convert.save_map("Shape_Generator", "row_node.osm", map)

                    def side_node_test():
                        
                        map = Convert.create_map()

                        for node_id, node in node_datas.items():
                            
                            (left_node, right_node) = A1.get_side(node_id)

                            if left_node != None:
                                line = [node.points[0], left_node.points[0]]
                                line3d = Convert.convert_to_lineString3d(line)
                                line3d.attributes["ID"] = str(node_id)
                                line3d.attributes["left_node"] = str(left_node.ID)
                                map.add(line3d)

                            if right_node != None:
                                line = [node.points[0], right_node.points[0]]
                                line3d = Convert.convert_to_lineString3d(line)
                                line3d.attributes["ID"] = str(node_id)
                                line3d.attributes["right_node"] = str(right_node.ID)
                                map.add(line3d)

                        Convert.save_map("Shape_Generator", "neighbor.osm", map)

                    node_datas = Shape.get_post_datas("A1")

                    # row_link_test()
                    # row_node_test()
                    # side_node_test()

                def save_endpoint():
                    
                    def save_all():
                        
                        map = Convert.create_map()

                        for node_id, node in Shape.get_post_datas("A1").items():
                            point = node.points[0]
                            endpoint = Endpoint.get_endpoint(node_id)
                            for side_index, side_p in enumerate(endpoint):
                                if side_p != None:
                                    line = [point, side_p]
                                    line3d = Convert.convert_to_lineString3d(line)
                                    line3d.attributes["ID"] = str(node_id)
                                    line3d.attributes["Side"] = "Left" if side_index == 0 else "Right"
                                    map.add(line3d)                        
                            
                        Convert.save_map("Shape_Generator", "endpoint({0}).osm".format(self.get_option()), map, sub_dir="endpoint/all")

                    def save_part():

                        if len(Shape.get_shape_datas("endpoint_part")) < 1:
                            Shape.set_shape_datas("endpoint_part", [{self.get_option() : {x : [None, None] for x in Shape.get_post_datas("A1").keys()}}])

                        map = Convert.create_map()

                        prev_record = Shape.get_shape_datas("endpoint_part")[-1].values()[0]

                        for node_id, node in Shape.get_post_datas("A1").items():
                            prev_endpoint = prev_record.get(node_id)
                            curr_endpoint = Endpoint.get_endpoint(node_id)
                            for side_index, side_p in enumerate(curr_endpoint):
                                if prev_endpoint[side_index] == None and curr_endpoint[side_index] != None:
                                    line = [node.points[0], side_p]
                                    line3d = Convert.convert_to_lineString3d(line)
                                    line3d.attributes["ID"] = str(node_id)
                                    line3d.attributes["Side"] = "Left" if side_index == 0 else "Right"
                                    map.add(line3d)

                        Convert.save_map("Shape_Generator", "endpoint({0} - part).osm".format(self.get_option()), map, sub_dir="endpoint/part")

                        Shape.get_shape_datas("endpoint_part").append({
                            self.get_option() : {x : Endpoint.get_endpoint(x) for x in Shape.get_post_datas("A1").keys()}
                            })  

                    save_all()
                    save_part()

                # - 각 A1 별
                # - 1) 생성 (절차 별 상이)
                # - 2) 갱신

                node_datas = Shape.get_post_datas("A1")
                _node_datas = {x[0] : x[1] for x in node_datas.items() if not Endpoint.check_endpoint(x[0])}

                counter = Process_Counter(len(_node_datas))

                for node_id in _node_datas.keys():

                    endpoint_before = Endpoint.get_endpoint(node_id)
                    endpoint_after = self._generate_endpoint(node_id)
                    
                    # - 갱신된 경우
                    if len([side_p for side_p in endpoint_before if side_p == None]) > len([side_p for side_p in endpoint_after if side_p == None]):
                        # 1) 현 A1 갱신 
                        Endpoint.set_endpoint(node_id, endpoint_after)
                        # 2) 좌/우 이웃 갱신
                        for side_index in [0, 1]:
                            if endpoint_before[side_index] != endpoint_after[side_index]:
                                self.update_neighbor(node_id, side_index)
                        
                        if Endpoint.check_endpoint(node_id):
                            counter.add(item="fix")

                    counter.add()
                    counter.print_sequence("[Endpoint] Generate endpoint ({0})".format(self.get_option()))
                counter.print_result("[Endpoint] Generate endpoint ({0})".format(self.get_option()))

                save_endpoint()

        class Generate_From_Neighbor(Generate_Process):
            """
            이웃 A1 중점 기반 생성 
            """

            def _generate_endpoint(self, node_id):
                
                endpoint = Endpoint.get_endpoint(node_id)

                for side_index in [0, 1]:
                    if endpoint[side_index] == None:
                        side_node = A1.get_side(node_id)[side_index]
                        if side_node != None:
                            node = Shape.get_post("A1", node_id)
                            side_p = get_mid(node.points[0], side_node.points[0])
                            endpoint[side_index] = side_p

                return endpoint

        class Generate_From_Lane(Generate_Process):
            """
            From / To A2 연관 B2 좌표 기반 생성
            """

            def _generate_endpoint(self, node_id):
                
                def check_range(horizontal, vertical, point):

                    max_vert = 0.25
                    # min_horz = 1.0 # 기본 
                    min_horz = 0.8 # KIAPI (폭이 좁은 경향)
                    max_horz = 10.0

                    vert_dist = calc_distance_from_line(horizontal, point)
                    if vert_dist < max_vert:
                        horz_dist = calc_distance_from_line(vertical, point)
                        if min_horz <= horz_dist <= max_horz:
                            return True
                            
                    return False 

                def get_intersect(node_id, horizontal, vertical, side_lanes):
                    """
                    A1 기반 수직선과 B2 교차점 중 최근접 추출
                    """

                    point = Shape.get_post("A1", node_id).points[0]

                    min_dist = float("inf")
                    min_p = None

                    for lane in side_lanes:
                        intersect_p = get_intersection_on_points(horizontal, lane.points)
                        if intersect_p != None:
                            if check_range(horizontal, vertical, intersect_p):
                                dist = calc_distance(point, intersect_p)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_p = intersect_p
                    
                    return min_p

                def get_close(node_id, horizontal, vertical, side_lanes):
                    
                    point = Shape.get_post("A1", node_id).points[0]

                    min_dist = float("inf")
                    min_p = None

                    lane_points = []
                    for side_lane in side_lanes:
                        lane_points += [side_lane.points[0], side_lane.points[-1]]
                    
                    for lane_point in lane_points:
                        if check_range(horizontal, vertical, lane_point):
                            dist = calc_distance(lane_point, point)
                            if dist < min_dist:
                                min_dist = dist
                                min_p = lane_point

                    return min_p

                endpoint = Endpoint.get_endpoint(node_id)

                # 1. From / To A2 추출
                (from_links, to_links) = A1.get_row_links(node_id)
                from_ids = [x.ID for x in from_links]
                to_ids = [x.ID for x in to_links]

                # 2. Left / Right B2 추출
                left_lanes = []
                right_lanes = []
                for link_id in (from_ids + to_ids):
                    (_left_lanes, _right_lanes) = A2.get_lanes(link_id)
                    left_lanes += _left_lanes
                    right_lanes += _right_lanes

                # 3. 수직 / 평행선 추출
                horizontal = self.get_horizontal(node_id)
                vertical = self.get_vertical(node_id)

                for side_index, side_lanes in enumerate([left_lanes, right_lanes]):
                    if endpoint[side_index] == None:
                        if len(side_lanes) > 0:
                            # 1) A1 기반 수직선과 B2 목록의 교차점 추출
                            side_p = get_intersect(node_id, horizontal, vertical, side_lanes)
                            # 2) 1번 X -> 근접 좌표 추출
                            if side_p == None:
                                side_p = get_close(node_id, horizontal, vertical, side_lanes)
                            # 3) 1, 2 번에서 추출 시 등록
                            if side_p != None:
                                endpoint[side_index] = side_p                                                                

                return endpoint

        class Generate_From_Row(Generate_Process):
            """
            From / To A1 너비 비례 기반 생성
            """

            def _generate_endpoint(self, node_id):

                def find_row(node_id, side_index, row_index):
                    
                    dist = None
                    length = None

                    node_stack = [node_id]
                    length_stack = [0.0]
                    record = defaultdict(lambda : False)
                    
                    while len(node_stack) > 0:
                        
                        # 1. 현 A1 의 너비 검사
                        curr_id = node_stack[-1]
                        curr_node = Shape.get_post("A1", curr_id)
                        side_p = Endpoint.get_endpoint(curr_id)[side_index]

                        # - 너비 추출 시
                        if side_p != None:
                            dist = calc_distance(curr_node.points[0], side_p)
                            length = sum(length_stack)
                            break                   

                        # 2. 다음 A1 추출
                        next_node = None
                        (from_nodes, to_nodes) = A1.get_row_nodes(curr_id)
                        for row_node in [from_nodes, to_nodes][row_index]:
                            if not record[row_node.ID]:
                                next_node = row_node

                        # - 다음 A1 추출 시 
                        if next_node != None:
                            # 1) stack 추가
                            node_stack.append(next_node.ID)
                            # 2) 발견 기록 (중복 검색 방지)
                            record[next_node.ID] = True
                            # 3) 현 A1 -> 다음 A1 까지의 거리를 stack 추가
                            (from_links, to_links) = A1.get_row_links(curr_id)
                            for row_link in [from_links, to_links][row_index]:
                                if getattr(row_link, ["FromNodeID", "ToNodeID"][row_index]) == next_node.ID:
                                    next_length = calc_length(row_link.points)
                                    length_stack.append(next_length)
                                    break 

                        # - 다음 A1 추출 실패 시 
                        else:
                            # - 1칸 롤백
                            node_stack.pop(-1)
                            length_stack.pop(-1) 

                    return dist, length

                endpoint = Endpoint.get_endpoint(node_id)

                for side_index in [0, 1]:
                    if endpoint[side_index] == None:
                        # 1. From / To 너비(dist) 및 거리(length) 추출 
                        from_dist, from_length = find_row(node_id, side_index, 0)
                        to_dist, to_length = find_row(node_id, side_index, 1)
                        if any([dist != None for dist in [from_dist, to_dist]]):
                            # - From / To 양측 추출 시 
                            if None not in [from_dist, to_dist]:
                                dist = (
                                    (from_dist * to_length / (from_length + to_length)) + 
                                    (to_dist * from_length / (from_length + to_length))
                                    )
                            # - From / To 한쪽 추출 시
                            else:
                                dist = from_dist if from_dist != None else to_dist

                            horizontal = self.get_horizontal(node_id, dist=dist)
                            endpoint[side_index] = horizontal[0] if side_index == 1 else horizontal[-1]                    

                return endpoint

        class Generate_From_Side(Generate_Process):
            """
            동일 A1 대칭 좌표 기반 생성
            """

            def _generate_endpoint(self, node_id):

                endpoint = Endpoint.get_endpoint(node_id)

                if any([side_p != None for side_p in endpoint]):
                    side_index = 0 if endpoint[0] == None else 1

                    point = Shape.get_post("A1", node_id).points[0]
                    vertical = self.get_vertical(node_id)
                    dist = calc_distance_from_line(vertical, endpoint[1-side_index])

                    side_p = get_ortho_line(point, start=vertical[0], end=vertical[-1], dist1=dist, dist2=dist)[0 if side_index == 1 else -1]
                    endpoint[side_index] = side_p

                return endpoint

        # --------------------------------------------------

        class Check_Process(Road_Process):

            options = {
                "Check_Undone" : "undone",
                "Check_Share" : "share",
                "Check_Reverse" : "reverse",
                "Check_Asymmetry" : "asymmetry",
            }

            def get_option(self):
                return self.__class__.options.get(self.__class__.__name__)

            # --------------------------------------------------

            @abstractmethod
            def _check_endpoint(self, node_id):
                pass

            # --------------------------------------------------

            def execute(self):

                node_datas = Shape.get_post_datas("A1")
                counter = Process_Counter(len(node_datas))

                record = {x : [True, True] for x in node_datas.keys()}

                for node_id in node_datas.keys():
                    record[node_id] = self._check_endpoint(node_id)
                    if False in record[node_id]:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Endpoint] Check process ({0})".format(self.get_option()))
                counter.print_result("[Endpoint] Check process ({0})".format(self.get_option()))

                self._save_check(record)

        class Check_Undone(Check_Process):
            
            def _check_endpoint(self, node_id):

                check = [True, True]

                endpoint = Endpoint.get_endpoint(node_id)
                for side_index, side_p in enumerate(endpoint):
                    if side_p == None:
                        check[side_index] = False

                return check

            def _save_check(self, record):

                map = Convert.create_map()

                for node_id, (left_check, right_check) in record.items():
                    if False in [left_check, right_check]:
                        point = Shape.get_post("A1", node_id).points[0]
                        point3d = Convert.convert_to_point3d(point)
                        point3d.attributes["ID"] = str(node_id)
                        for side_index, check in enumerate([left_check, right_check]):
                            if not check:
                                point3d.attributes["{0}".format("Left" if side_index == 0 else "Right")] = ""
                        map.add(point3d)        

                Convert.save_map("Shape_Generator", "endpoint_check({0}).osm".format(self.get_option()), map, sub_dir="endpoint/check")

        class Check_Share(Check_Process):

            def _check_endpoint(self, node_id):
                
                check = [True, True]

                endpoint = Endpoint.get_endpoint(node_id)

                (left_node, right_node) = A1.get_side(node_id)
                if left_node != None:
                    if endpoint[0] != Endpoint.get_endpoint(left_node.ID)[1]:
                        check[0] = False
                if right_node != None:
                    if endpoint[1] != Endpoint.get_endpoint(right_node.ID)[0]:
                        check[1] = False

                return check

            def _save_check(self, record):

                map = Convert.create_map()

                for node_id, (left_check, right_check) in record.items():
                    if False in [left_check, right_check]:
                        point = Shape.get_post("A1", node_id).points[0]
                        point3d = Convert.convert_to_point3d(point)
                        point3d.attributes["ID"] = str(node_id)
                        for side_index, check in enumerate([left_check, right_check]):
                            if not check:
                                point3d.attributes["{0}".format("Left" if side_index == 0 else "Right")] = ""
                        map.add(point3d)    

                Convert.save_map("Shape_Generator", "endpoint_check({0}).osm".format(self.get_option()), map, sub_dir="endpoint/check")

        class Check_Reverse(Check_Process):
            """
            좌/우 뒤집힌 Endpoint 검출
            - A2 시작 좌표가 중첩된 경우 발생하기 쉬움 (A2_Parser - parse_points)
            """

            def _check_endpoint(self, node_id):

                check = [True, True]

                endpoint = Endpoint.get_endpoint(node_id)
                vertical = self.get_vertical(node_id)

                if None not in endpoint:
                    left_check = check_is_left(vertical[0], vertical[-1], endpoint[0])
                    right_check = not check_is_left(vertical[0], vertical[-1], endpoint[1])
                    if not left_check:
                        check[0] = False
                    if not right_check:
                        check[1] = False

                return check

            def _save_check(self, record):

                map = Convert.create_map()

                for node_id, (left_check, right_check) in record.items():
                    if False in [left_check, right_check]:
                        point = Shape.get_post("A1", node_id).points[0]
                        endpoint = Endpoint.get_endpoint(node_id)

                        for side_index, side_p in enumerate(endpoint):
                            line = [point, side_p]
                            line3d = Convert.convert_to_lineString3d(line)
                            line3d.attributes["ID"] = str(node_id)
                            line3d.attributes["side"] = str(side_index)
                            map.add(line3d)

                Convert.save_map("Shape_Generator", "endpoint_check({0}).osm".format(self.get_option()), map, sub_dir="endpoint/check")

        class Check_Asymmetry(Check_Process):
            """
            좌/우 뒤집힌 Endpoint 검출
            - A2 시작 좌표가 중첩된 경우 발생하기 쉬움 (A2_Parser - parse_points)
            """

            def _check_endpoint(self, node_id):

                check = [True, True]

                endpoint = Endpoint.get_endpoint(node_id)
                vertical = self.get_vertical(node_id)

                if None not in endpoint:
                    left_check = check_is_left(vertical[0], vertical[-1], endpoint[0])
                    right_check = not check_is_left(vertical[0], vertical[-1], endpoint[1])
                    if left_check != right_check:
                        check = [False, False]

                return check 

            def _save_check(self, record):

                map = Convert.create_map()

                for node_id, (left_check, right_check) in record.items():
                    if False in [left_check, right_check]:
                        point = Shape.get_post("A1", node_id).points[0]
                        endpoint = Endpoint.get_endpoint(node_id)

                        for side_index, side_p in enumerate(endpoint):
                            line = [point, side_p]
                            line3d = Convert.convert_to_lineString3d(line)
                            line3d.attributes["ID"] = str(node_id)
                            line3d.attributes["side"] = str(side_index)
                            map.add(line3d)

                Convert.save_map("Shape_Generator", "endpoint_check({0}).osm".format(self.get_option()), map, sub_dir="endpoint/check")

        # --------------------------------------------------

        def execute(self):

            self.Generate_From_Lane().execute()
            self.Generate_From_Neighbor().execute()
            self.Generate_From_Row().execute()
            self.Generate_From_Side().execute()

            self.Check_Undone().execute()
            self.Check_Share().execute()
            self.Check_Reverse().execute()
            self.Check_Asymmetry().execute()

    # --------------------------------------------------
        
    class Bound_Generator():

        class Generate_From_Lane(Road_Process):

            def share_side(self, link_id, side_index):

                link = Shape.get_post("A2", link_id)
                side_id = getattr(link, ["L_LinkID", "R_LinkID"][side_index])
                
                side_bound = Bound.get_bound(side_id)
                side = side_bound[1-side_index]

                return side

            def create_side(self, link, side_index):
                """
                좌/우(side_index) bound 에 해당하는 좌표목록 생성 
                """

                def create_parts(link, side_index, link_points):
                    """
                    Lane 에서 좌/우(side_index) bound 에 해당하는 좌표 추출
                    - 현 A2 를 등록한 B2 목록에서 좌표 추출
                    - B2 목록 중 가상 차선(B2.Type == 999) 는 추출 대상에서 제외
                    """

                    # 추출 좌표 목록 (2D list)
                    bound_parts = []

                    from_id = link.FromNodeID
                    to_id = link.ToNodeID

                    # 1. bound 의 시작/종료 좌표 추출 (endpoint)
                    s_point = Endpoint.get_endpoint(from_id)[side_index]
                    e_point = Endpoint.get_endpoint(to_id)[side_index]

                    # 2. 추출 좌표 목록에 시작점 등록
                    point_data = {
                        "index" : 0,
                        "point" : s_point 
                    }
                    bound_parts.append([point_data])

                    # 3. B2 목록 추출
                    (left_lanes, right_lanes) = A2.get_lanes(link.ID)
                    side_lanes = [left_lanes, right_lanes][side_index]
                    # - 가상 차선 제외
                    side_lanes = [x for x in side_lanes if x.Type != 999]

                    map = Convert.create_map()

                    if len(side_lanes) > 0:

                        if link.ID in ["A219BS010229_0"] and side_index == 1:
                            map.add(Convert.convert_to_lineString3d(link_points))

                        for index in range(1, len(link_points) - 1):
                            
                            curr_p = link_points[index]
                            next_p = link_points[index+1]
                            ortho_line = get_ortho_line(curr_p, start=curr_p, end=next_p)

                            if link.ID in ["A219BS010229_0"] and side_index == 1:
                                map.add(Convert.convert_to_lineString3d(ortho_line))

                            intersect_list = []
                            for lane in side_lanes:
                                lane_points = lane.points
                                intersect_p = get_intersection_on_points(ortho_line, lane_points)
                                if intersect_p != None:
                                    intersect_list.append(intersect_p)

                            if len(intersect_list) > 0:
                                intersect_p = sorted(intersect_list, key=lambda x : calc_distance(curr_p, x))[0]
                                point_data = {
                                    "index" : index,
                                    "point" : intersect_p,
                                }
                                bound_parts[-1].append(point_data)
                            else:
                                if len(bound_parts[-1]) > 0:
                                    bound_parts.append([])
                    else:
                        bound_parts.append([])
                    
                    point_data = {
                        "index" : len(link_points) - 1,
                        "point" : e_point,
                    }
                    bound_parts[-1].append(point_data)

                    if link.ID in ["A219BS010229_0"] and side_index == 1:
                        Convert.save_map("Shape_Generator", "test.osm", map)

                    return bound_parts

                def interpolate_parts(link, side_index, link_points, bound_parts):
                    
                    def calc_start_width(link, curr_part, curr_points, is_start=False):
                        
                        link_point = curr_points[0]
                        lane_point = curr_part[-1]["point"]

                        if is_start and len(curr_part) <= 1:
                            vertical = self.get_vertical(link.FromNodeID)
                            start_width = calc_distance_from_line(vertical, lane_point)
                        else:
                            start_width = calc_distance(link_point, lane_point)

                        return start_width

                    def calc_end_width(link, next_part, curr_points, is_end=False):

                        link_point = curr_points[-1]
                        lane_point = next_part[0]["point"]

                        if is_end and len(next_part) <= 1:
                            vertical = self.get_vertical(link.ToNodeID)
                            end_width = calc_distance_from_line(vertical, lane_point)
                        else:
                            end_width = calc_distance(link_point, lane_point)

                        return end_width

                    def calc_curr_width(prev_width, e_width, link_points, link_index):
                        """
                        현 너비 = 전 너비 + 전 너비차(= 종료 너비 - 전 너비) X (이전좌표 ~ 현좌표 / 남은 길이)
                        """
                        
                        prev_interval = e_width - prev_width
                        prev_length = calc_length(link_points[link_index-1:link_index+1])
                        remain_length = calc_length(link_points[link_index-1:])
                        curr_width = prev_width + prev_interval * (prev_length / remain_length)

                        return curr_width

                    side = []

                    # - 보간 필요 시
                    if len(bound_parts) > 1:
                        for part_index in range(len(bound_parts) - 1):
                            
                            curr_part = bound_parts[part_index]
                            next_part = bound_parts[part_index+1]

                            # 1. 보간 영역 (A2) 추출
                            s_index = curr_part[-1]["index"]
                            e_index = next_part[0]["index"]
                            curr_points = link_points[s_index:e_index+1]

                            # 2. 보간 시작/종료 너비 추출
                            s_width = calc_start_width(link, curr_part, curr_points, is_start=part_index==0)
                            e_width = calc_end_width(link, next_part, curr_points, is_end=part_index==len(bound_parts)-2)
                            prev_width = s_width

                            # 3. 보간
                            interp_points = []

                            for link_index in range(1, len(curr_points) - 1):
                                # 1) 현 너비 추출
                                curr_width = calc_curr_width(prev_width, e_width, curr_points, link_index)
                                # 2) 현 좌표기반 수직선 추출 (너비 지정)
                                direction = curr_points[link_index:link_index+2]
                                # 3) 보간 좌표 추출 (수직선 상)
                                ortho_line = get_ortho_line(curr_points[link_index], start=direction[0], end=direction[-1], dist1=curr_width, dist2=curr_width)
                                curr_point = ortho_line[0 if side_index == 1 else -1] 
                                # 4) 보간 좌표 추가
                                interp_points.append(curr_point)
                                prev_width = curr_width
                            
                            # 4. 전 영역(B2 추출) 추가
                            for lane_point in [data["point"] for data in curr_part]:
                                side.append(lane_point)

                            # 5. 보간 영역 추가
                            for interp_point in interp_points:
                                side.append(interp_point)
                        # - 모든 영역 사이의 보간 종료 시
                        else:
                            # 6. 마지막 영역(B2 추출) 추가
                            for lane_point in [data["point"] for data in next_part]:
                                side.append(lane_point)     
                    # - 보간 불필요 시 
                    else:
                        # - 1개의 통합된 B2 영역 추가
                        for lane_point in [data["point"] for data in bound_parts[0]]:
                            side.append(lane_point)

                    return side

                def filter_side(side):

                    def check_close(point, side, close_dist=1.0):

                        if calc_distance(point, side[0]) <= close_dist:
                            return True
                        elif calc_distance(point, side[-1]) <= close_dist:
                            return True
                        
                        return False

                    _side = side[:1]

                    for index in range(1, len(side) - 1):
                        point = side[index]
                        if not check_close(point, side):
                            _side.append(point)
                    else:
                        _side.append(side[-1])

                    return _side

                # 1. A2 밀도 증가 (추출 정밀도 증가)
                link_points = improve_points_density(link.points)
                # 2. B2 기반 좌표 추출 = 부분영역 추출
                bound_parts = create_parts(link, side_index, link_points)
                # 3. 부분영역 기반 보간
                side = interpolate_parts(link, side_index, link_points, bound_parts)
                # 4. 선 단순화
                side = simplify_polygon(side)
                # 5. 방향 오류 제거
                side = correct_misdirected(side)
                # 6. 시작/종료 근접 필터링
                side = filter_side(side)

                return side

            def create_centerLine(self, left, right):

                centerLine = []

                # - 좌/우 bound 중 긴 길이를 1m 로 나눈 개수
                lengths = [calc_length(x) for x in [left, right]]
                sample_count = int(max(lengths))

                # - 좌/우 bound 샘플링 
                left_samples = sample_points(left, count=sample_count)
                right_samples = sample_points(right, count=sample_count)

                # - 중심선 생성
                for index in range(sample_count + 1):
                    left_p = left_samples[index] 
                    right_p = right_samples[index]
                    mid_p = get_mid(left_p, right_p)
                    centerLine.append(mid_p)

                return centerLine

            def smooth_centerLine(self, link, centerLine, is_merge, is_branch):
                
                centerLine = improve_points_density(centerLine)
                link_points = improve_points_density(link.points)

                if is_merge and is_branch:
                    pass
                elif is_merge:
                    for i in range(len(link_points)):
                        point = link_points[i]
                        closest_p = get_closest_point(point, centerLine)
                        if calc_distance(point, closest_p) < 0.5:
                            i = i + 1 if i < 1 else i
                            j = centerLine.index(closest_p)
                            centerLine = link_points[:i] + centerLine[j:]
                            centerLine = simplify_polygon(centerLine[:i+1], elapse_dist=0.5) + centerLine[i+1:]                                  
                            break
                    else:
                        centerLine = link_points[:len(link_points)/2] + centerLine[len(centerLine)/2:]
                elif is_branch:
                    for i in range(len(link_points) - 1, -1, -1):
                        point = link_points[i]
                        closest_p = get_closest_point(point, centerLine)
                        if calc_distance(point, closest_p) < 0.5:
                            i = i -1 if i >= len(link_points) - 1 else i
                            j = centerLine.index(closest_p)
                            centerLine = centerLine[:j] + link_points[i:]                                 
                            centerLine = centerLine[:j] + simplify_polygon(centerLine[j:], elapse_dist=0.5)                                 
                            break
                    else:
                        centerLine = centerLine[:len(centerLine)/2] + link_points[len(link_points)/2:]

                centerLine = simplify_polygon(centerLine)
                
                return centerLine

            def generate_bound(self, link_id):

                def generate_sides(link):

                    sides = [None, None]

                    # 1. 좌/우 bound 생성
                    for side_index in [0, 1]:
                        side = self.share_side(link.ID, side_index)
                        if side == None:
                            # - 병합 A2 에서 Sub 에 해당하는 경우 : 현재 방향이 Main 방향이라면 bound 생성 금지 (공유만 허용)
                            if A2.check_merge(link.ID, side_index) or A2.check_branch(link.ID, side_index):
                                side_link = A2.get_side(link)[side_index]
                                side = self.create_side(side_link, 1-side_index)
                            else:
                                # - 현 A2 기반 bound 생성
                                side = self.create_side(link, side_index)
                        sides[side_index] = side

                    return sides

                def generate_center(link, left, right):

                    centerLine = self.create_centerLine(left, right)
                    centerLine = simplify_polygon(centerLine)

                    return centerLine

                left = None
                right = None
                center = None

                link = Shape.get_post("A2", link_id)

                (left, right) = generate_sides(link)
                center = generate_center(link, left, right)

                bound = [left, right, center]
                Bound.set_bound(link_id, bound)

            def generate_bounds(self):

                link_datas = Shape.get_post_datas("A2")
                counter = Process_Counter(len(link_datas))

                # - 차로번호(LaneNo)가 낮은 순으로 생성
                # - 동일한 차로번호의 경우 길이가 긴 순으로 생성
                sorted_keys = sorted(link_datas.keys(), key=lambda x : (
                    link_datas.get(x).LaneNo,
                    -link_datas.get(x).Length, 
                    ))

                for link_id in sorted_keys:
                    self.generate_bound(link_id)
                    counter.add()
                    counter.print_sequence("[Bound] Generate bound")
                counter.print_result("[Bound] Generate bound")

            # --------------------------------------------------

            def execute(self):

                def save_bounds():

                    map = Convert.create_map()

                    link_datas = Shape.get_post_datas("A2")
                    record = defaultdict(lambda : [None, None])

                    for link_id, link in link_datas.items():
                        (left_link, right_link) = A2.get_side(link)
                        for side_index, side_link in enumerate([left_link, right_link]):
                            line3d = None
                            if side_link != None:
                                side_id = side_link.ID
                                line3d = record[side_id][1-side_index]
                            if line3d == None:
                                line = Bound.get_bound(link_id)[side_index]
                                line3d = Convert.convert_to_lineString3d(line)
                            line3d.attributes["{0}".format("Left" if side_index == 0 else "Right")] = str(link_id)
                            map.add(line3d)
                            record[link_id][side_index] = line3d

                    Convert.save_map("Shape_Generator", "bound.osm", map, sub_dir="bound")

                self.generate_bounds()

                save_bounds()

        # --------------------------------------------------

        @add_metaclass(ABCMeta)
        class Check_Process():

            options = {
                "Check_Undone" : "undone",
                "Check_Endpoint" : "endpoint",
            }

            def get_option(self):
                return self.__class__.options.get(self.__class__.__name__)

            def execute(self):
                
                link_datas = Shape.get_post_datas("A2")
                counter = Process_Counter(len(link_datas))

                record = defaultdict(lambda : [True, True])

                for link_id in link_datas.keys():
                    record[link_id] = self._check_bound(link_id)
                    if not all([check == None for check in record[link_id]]):
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Bound] Check bound ({0})".format(self.get_option()))
                counter.print_result("[Bound] Check bound ({0})".format(self.get_option()))

                self._save(record)

            # --------------------------------------------------

            @abstractmethod
            def _check_bound(self, link_id):
                pass

            @abstractmethod
            def _save(self, record):
                pass

        class Check_Undone(Check_Process):

            def _check_bound(self, link_id):

                check = [None, None]

                (left, right, center) = Bound.get_bound(link_id)
                for side_index, side in enumerate([left, right]):
                    if side == None:
                        check[side_index] = False

                return check

            def _save(self, record):
                
                map = Convert.create_map()

                for link_id, [left_check, right_check] in record.items():
                    if False in [left_check, right_check]:
                        line = Shape.get_post("A2", link_id).points
                        line3d = Convert.convert_to_lineString3d(line)
                        line3d.attributes["ID"] = str(link_id)
                        for side_index, check in enumerate([left_check, right_check]): 
                            if not check:
                                line3d.attributes["{0}".format("Left" if side_index == 0 else "Right")] = ""
                        map.add(line3d)

                Convert.save_map("Shape_Generator", "{0}.osm".format(self.get_option()), map, sub_dir="bound/check")

        class Check_Endpoint(Check_Process):
            """
            Bound 시작/종료 좌표가 Endpoint 와 일치하는 여부 검사
            - Bound share 시 발생 경향
            """

            def _check_bound(self, link_id):

                check = [None, None]

                link = Shape.get_post("A2", link_id)

                for row_index in [0, -1]:
                    node_id = getattr(link, ["FromNodeID", "ToNodeID"][row_index])
                    endpoint = Endpoint.get_endpoint(node_id)
                    for side_index in [0, 1]:
                        if row_index == 0 and A2.check_branch(link_id, side_index):
                            continue
                        if row_index == -1 and A2.check_merge(link_id, side_index):
                            continue
                        side_p = endpoint[side_index]
                        side = Bound.get_bound(link.ID)[side_index]
                        if not check_same(side_p, side[row_index]):
                            if check[side_index] == None:
                                check[side_index] = [True, True]
                            check[side_index][row_index] = False                            

                return check

            def _save(self, record):
                
                map = Convert.create_map()

                for link_id, [left_check, right_check] in record.items():
                    for side_index, side_check in enumerate([left_check, right_check]): 
                        if side_check != None:
                            side = Bound.get_bound(link_id)[side_index]
                            for row_index in [0, -1]:
                                link = Shape.get_post("A2", link_id)
                                node = Shape.get_post("A1", getattr(link, ["FromNodeID", "ToNodeID"][row_index]))
                                endpoint = Endpoint.get_endpoint(node.ID)
                                side_p = endpoint[side_index]
                                row_check = side_check[row_index]
                                if not row_check:
                                    line = [side_p, side[row_index]]
                                    line3d = Convert.convert_to_lineString3d(line) 
                                    line3d.attributes["ID"] = str(link_id)
                                    line3d.attributes["Side"] = "Left" if side_index == 0 else "Right"
                                    line3d.attributes["Row"] = "From" if row_index == 0 else "To"
                                    map.add(line3d)

                Convert.save_map("Shape_Generator", "{0}.osm".format(self.get_option()), map, sub_dir="bound/check")

        # --------------------------------------------------

        def execute(self):
            self.Generate_From_Lane().execute()
            self.Check_Undone().execute()
            self.Check_Endpoint().execute()

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.Endpoint_Generator().execute()
        cls.Bound_Generator().execute()


class Crosswalk_Generator():

    class Bound_Generator():

        class Generate_From_Polygon():
            
            def execute(self):
                
                def split(points):
                    
                    def get_index(index, count):
                        _index = index % count
                        return _index

                    # 1. 다각형을 사각형으로 변형
                    rectangle = reduce_polygon(points, 4)

                    # 2. 가장 긴 변 추출
                    max_dist = -1
                    max_index = -1
                    for index in range(len(rectangle)):
                        A = rectangle[index]
                        B = rectangle[index+1] if index < len(rectangle) - 1 else rectangle[0]
                        dist = calc_distance(A, B)
                        if dist > max_dist:
                            max_dist = dist
                            max_index = index

                    # 3. 좌/우 변의 시작, 종료 목차 추출
                    left_s = get_index(max_index + 1, 4)
                    left_e = get_index(max_index, 4)
                    right_s = get_index(max_index + 3, 4)
                    right_e = get_index(max_index + 2, 4)

                    # 4. 시작, 종료 목차 사이의 좌표들로 변 추출 
                    # 4-1) 좌변 추출
                    left = []
                    left_s = points.index(get_closest_point(rectangle[left_s], points))
                    left_e = points.index(get_closest_point(rectangle[left_e], points))
                    index = left_s
                    while True:
                        point = points[index]
                        left.append(point)
                        if index == left_e:
                            break
                        index = index + 1
                        index %= len(points)
                    # 4-1) 우변 추출
                    right = []
                    right_s = points.index(get_closest_point(rectangle[right_s], points))
                    right_e = points.index(get_closest_point(rectangle[right_e], points))
                    index = right_s
                    while True:
                        point = points[index]
                        right.append(point)
                        if index == right_e:
                            break
                        index = index + 1
                        index %= len(points)

                    return (left, right)

                mark_datas = Shape.get_post_datas("B3")
                counter = Process_Counter(len(mark_datas))

                for mark_id, mark in mark_datas.items():
                    if Crosswalk.check_crosswalk(mark_id):
                        (left, right) = split(mark.points)
                        Crosswalk.set_bound(mark_id, [left, right])
                        counter.add(item="fix")
                    counter.add()
                    counter.print_sequence("[Crosswalk] Generate crosswalk")
                counter.print_result("[Crosswalk] Generate crosswalk")

        # --------------------------------------------------

        def execute(self):

            def save_origin():

                map = Convert.create_map()
                mark_datas = Shape.get_shape_datas("B3")

                for mark_id, mark in mark_datas.items():
                    if Crosswalk.check_crosswalk(mark_id):
                        points = simplify_polygon(mark.points)
                        map.add(Convert.convert_to_lineString3d(points))
                
                Convert.save_map("Shape_Generator", "origin.osm", map, sub_dir="Crosswalk")

            def save_crosswalk():

                map = Convert.create_map()

                mark_datas = Shape.get_shape_datas("B3")
                for mark_id, mark in mark_datas.items():
                    if Crosswalk.check_crosswalk(mark_id):
                        (left, right) = Crosswalk.get_bound(mark_id)
                        lanelet = Convert.convert_to_lanelet(left, right)
                        lanelet.attributes["B3"] = mark_id
                        map.add(lanelet)

                Convert.save_map("Shape_Generator", "crosswalk.osm", map, sub_dir="Crosswalk")

            self.Generate_From_Polygon().execute()

            save_origin()
            save_crosswalk()

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.Bound_Generator().execute()


class Regulatory_Generator():

    class Direction():

        class Light_Direction():
    
            class Vehicle():
                
                @classmethod
                def get_direction(cls, light):
                    """
                    # 1) 주행등 2개이상 : 신호등 간 직선방향 추출
                    # 2) 주행등 2개미만 : 신호등 - 지주 간 직선방향 추출
                    """

                    def except_lights(lights):
                        
                        _lights = []
                        for light in lights:
                            if light.Type not in [11, 12, 13]:
                                _lights.append(light)

                        return _lights

                    direction = None

                    post = Shape.get_shape("C6", light.PostID)
                    if post != None:
                        lights = Light.get_lights(post_id=post.ID)
                        lights = except_lights(lights)
                        # 1) 주행등 2개 이상 : 주행등 간 직선방향 추출
                        if len(lights) > 1:
                            # - 신호지주가 주행경로의 우측에 위치하는 것을 전제로 함 (우측 통행 기준)
                            lights = sorted(lights, key=lambda x : calc_distance(post.points[0], x.points[0]))
                            direction = [lights[-1].points[0], lights[0].points[0]]
                        # 2) 주행등 2개 미만 : 신호지주 기반 방향 추출
                        else:
                            # - 기본적으로 도로 진행방향 기준 신호등이 좌측, 지주가 우측인 것을 가정
                            direction = [light.points[0], post.points[0]]
                            # - 지주가 좌측, 신호등이 우측에 해당하는 경우 방향을 뒤집는다.
                            link = Light.get_link(light.ID)
                            if link != None:
                                if check_is_left(light.points[0], post.points[0], link.points[0]):
                                    direction = direction[::-1]
                    # - 신호지주 없는 경우
                    else:
                        # 3) 주행경로 참조
                        link = Light.get_link(light.ID)
                        if link != None:
                            # - 우측통행 기준 (주행경로 시계방향 90도 회전 = 신호등 방향)
                            seg = get_closest_segment(light.points[0], link.points)
                            direction = rotate_seg(seg, deg=90)

                    return direction

            class Pedestrian():

                @classmethod
                def get_direction(cls, light):
                    
                    direction = None

                    link = Light.get_link(light.ID)
                    if link != None:
                        # - 우측통행 기준 (주행경로 시계방향 90도 회전 = 신호등 방향)
                        seg = get_closest_segment(light.points[0], link.points)
                        direction = rotate_seg(seg, deg=-90)
                    else:
                        quad_tree, quad_table = Quad.get_quad("A2")
                        closest_p = get_closest_quad_point(light.points[0], quad_tree)
                        if closest_p != None:
                            point2d = (closest_p[0], closest_p[1])
                            link_ids = quad_table[point2d]
                            for link_id in link_ids:
                                link = Shape.get_shape("A2", link_id)
                                if link != None:
                                    seg = get_closest_segment(light.points[0], link.points)
                                    direction = rotate_seg(seg, deg=-90)
                                    break
                    
                    return direction

            # ---------------------------------------------

            def get_direction(self, light):

                def get_instance(light):
                    """
                    주행등 / 보행등 구분
                    """
                    code = light.Type
                    if code not in [11, 12, 13]:
                        return self.Vehicle
                    else:
                        return self.Pedestrian

                # - 주행등 
                template = get_instance(light)
                direction = template.get_direction(light)
                return direction

        class Sign_Direction():
        
            def get_direction(self, sign):
                """
                1) 주행경로 참조
                2) 신호지주 참조
                """

                direction = None
            
                # 1) 주행경로 참조
                link = Light.get_link(sign.ID)
                if link != None:
                    # - 우측통행 기준 (주행경로 시계방향 90도 회전 = 신호등 방향)
                    seg = get_closest_segment(sign.points[0], link.points)
                    direction = rotate_seg(seg, deg=90)
                else:
                    quad_tree, quad_table = Quad.get_quad("A2")
                    closest_p = get_closest_quad_point(sign.points[0], quad_tree)
                    if closest_p != None:
                        point2d = (closest_p[0], closest_p[1])
                        link_ids = quad_table[point2d]
                        for link_id in link_ids:
                            link = Shape.get_shape("A2", link_id)
                            if link != None:
                                seg = get_closest_segment(sign.points[0], link.points)
                                direction = rotate_seg(seg, deg=-90)
                                break

                return direction

        @classmethod
        def get_direction(cls, shape):
            
            if shape.ID[:2] == "C1":
                direction = cls.Light_Direction().get_direction(shape)
            else:
                direction = cls.Sign_Direction().get_direction(shape)

            return direction

    # --------------------------------------------------

    class StopLine_Generator():

        def __init__(self):
            
            def create_stopLine_datas():

                stopLine_datas = dict()
                lane_datas = Shape.get_shape_datas("B2")
                
                counter = Process_Counter(len(lane_datas))

                for lane_id, lane in lane_datas.items():
                    if lane.Kind == 530:
                        stopLine_datas[lane_id] = lane.points
                        counter.add(item="fix")
                    counter.add()
                    counter.print_sequence("[StopLine] Create stopLine datas")
                counter.print_result("[StopLine] Create stopLine datas")

                setattr(self.__class__, "stopLine_datas", stopLine_datas)

            def create_link_datas():

                link_datas = dict()
                
                light_datas = Shape.get_shape_datas("C1")
                sign_datas = Shape.get_shape_datas("B1")

                counter = Process_Counter(len(light_datas) + len(sign_datas))

                for light in light_datas.values():
                    link = Shape.get_shape("A2", light.LinkID)
                    if link != None:
                        link_datas[link.ID] = link
                        counter.add(item="fix")
                    counter.add()
                    counter.print_sequence("[StopLine] Create link datas")
                
                for sign in sign_datas.values():
                    link = Shape.get_shape("A2", sign.LinkID)
                    if link != None:
                        link_datas[link.ID] = link
                        counter.add(item="fix")
                    counter.add()
                    counter.print_sequence("[StopLine] Create link datas")

                counter.print_result("[StopLine] Create link datas")

                setattr(self.__class__, "link_datas", link_datas)

            create_stopLine_datas()
            create_link_datas()

        # --------------------------------------------------

        @add_metaclass(ABCMeta)
        class Generate_Process():

            def get_option(self):
                return self.__class__.__name__.split("_")[-1]

            # --------------------------------------------------

            def create_stopLine_table(self):

                stopLine_datas = getattr(Regulatory_Generator.StopLine_Generator, "stopLine_datas") 
                counter = Process_Counter(len(stopLine_datas))

                process_datas = dict()

                for lane_id, points in stopLine_datas.items():
                    lanelet_keys = self.find_links(points)
                    if len(lanelet_keys) > 0:
                        for lanelet_key in lanelet_keys:
                            StopLine.set_key(lanelet_key, lane_id)
                            self.update_neighbor(lanelet_key)
                            process_datas[lanelet_key] = lane_id
                            counter.add(item="fix")
                    counter.add()
                    counter.print_sequence("[StopLine : {0}] Generate stopLine table".format(self.get_option()))
                counter.print_result("[StopLine : {0}] Generate stopLine table".format(self.get_option()))

                setattr(Regulatory_Generator.StopLine_Generator, "{0}_datas".format(self.get_option()), process_datas)

            def update_neighbor(self, lanelet_key):
                
                stopLine_key = StopLine.get_key(lanelet_key)

                if stopLine_key != None:
                    link = Shape.get_shape("A2", lanelet_key)
                    for side_index in [0, 1]:
                        side_id = getattr(link, ["L_LinkID", "R_LinkID"][side_index])
                        side_link = Shape.get_shape("A2", side_id)
                        if side_link != None:
                            if StopLine.get_key(side_id) == None:
                                StopLine.set_key(side_id, stopLine_key)

            # --------------------------------------------------

            @abstractmethod
            def find_links(self, lane):
                pass

            # --------------------------------------------------

            def execute(self):

                def save_table():
                    """
                    A2 - 정지선 관계 맵
                    - 이전 단계에서 추출된 관계 누적
                    """

                    link_datas = getattr(Regulatory_Generator.StopLine_Generator, "link_datas")

                    map = Convert.create_map()

                    for link_id in link_datas.keys():
                        stopLine_key = StopLine.get_key(link_id)
                        if stopLine_key != None:
                            stopLine = StopLine.get_stopLine(stopLine_key=stopLine_key)
                            line3d = Convert.convert_to_lineString3d(stopLine)
                            line3d.attributes["Link"] = str(link_id)
                            map.add(line3d)
                            link = link_datas.get(link_id)
                            line3d = Convert.convert_to_lineString3d(link.points)
                            line3d.attributes["ID"] = str(link_id)
                            map.add(line3d)

                    Convert.save_map("Shape_Generator", "stopLine_table({0}).osm".format(self.__class__.__name__), map, sub_dir="Regulatory/StopLine/{0}".format(self.get_option()))

                def save_undone():

                    stopLine_datas = getattr(Regulatory_Generator.StopLine_Generator, "stopLine_datas")
                    link_datas = getattr(Regulatory_Generator.StopLine_Generator, "link_datas")

                    done_link_datas = dict()
                    done_stopLine_datas = dict()

                    # 1. 등록 데이터 추출
                    for link_id, link in link_datas.items():
                        stopLine_key = StopLine.get_key(link_id)
                        if stopLine_key != None:
                            # 1) 등록 A2 기록
                            done_link_datas[link_id] = True
                            # 2) 등록 StopLine 기록
                            done_stopLine_datas[stopLine_key] = True

                    # 2. 미등록 A2 맵
                    map = Convert.create_map()
                    for link_id, link in link_datas.items():
                        if done_link_datas.get(link_id) == None:
                            line3d = Convert.convert_to_lineString3d(link.points)
                            map.add(line3d)
                    Convert.save_map("Shape_Generator", "undone_link.osm", map, sub_dir="Regulatory/StopLine/{0}".format(self.get_option()))

                    # 3. 미등록 StopLine 맵
                    map = Convert.create_map()
                    for stopLine_key, stopLine in stopLine_datas.items():
                        if done_stopLine_datas.get(stopLine_key) == None:
                            line3d = Convert.convert_to_lineString3d(stopLine)
                            map.add(line3d)
                    Convert.save_map("Shape_Generator", "undone_stopLine.osm", map, sub_dir="Regulatory/StopLine/{0}".format(self.get_option()))

                self.create_stopLine_table()

                save_table()
                save_undone()

        class Generate_From_End(Generate_Process):
            """
            A2 종료 좌표에 근접한 정지선 등록
            """

            def find_links(self, points):
                
                def check_close(origin, point):
                    if calc_distance(origin, point) <= 1:
                        return True
                    return False

                link_ids = []

                (quad_tree, quad_table) = Quad.get_quad("A2_end")

                # - 1m 간격 좌표 배치
                stopLine = improve_points_density([points[0], points[-1]])

                for point in stopLine:

                    _link_ids = []
                    closest_p = get_closest_quad_point(point, quad_tree)

                    if closest_p == None:
                        break
                
                    if check_close(point, closest_p):
                        point2d = (closest_p[0], closest_p[1])
                        for link_id in quad_table[point2d]:
                            # - 이미 정지선이 등록된 A2 제외
                            if StopLine.get_stopLine(link_id=link_id) == None:
                                _link_ids.append(link_id)

                    link_ids += _link_ids

                # - 중복 제거
                link_ids = list(set(link_ids))

                return link_ids

        class Generate_From_Start(Generate_Process):
            """
            A2 시작 좌표에 근접한 정지선 등록
            """

            def find_links(self, points):
                
                def check_close(origin, point):
                    if calc_distance(origin, point) <= 1:
                        return True
                    return False

                link_ids = []

                (quad_tree, quad_table) = Quad.get_quad("A2_start")

                # - 1m 간격 좌표 배치
                stopLine = improve_points_density([points[0], points[-1]])

                for point in stopLine:

                    _link_ids = []
                    closest_p = get_closest_quad_point(point, quad_tree)

                    if closest_p == None:
                        break
                
                    if check_close(point, closest_p):
                        point2d = (closest_p[0], closest_p[1])
                        for link_id in quad_table[point2d]:
                            # - 이미 정지선이 등록된 A2 제외
                            if StopLine.get_stopLine(link_id=link_id) == None:
                                _link_ids.append(link_id)

                    link_ids += _link_ids

                # - 중복 제거
                link_ids = list(set(link_ids))

                return link_ids

        class Generate_From_Intersect(Generate_Process):

            def find_links(self, points):
                
                def check_close(origin, point):
                    if calc_distance(origin, point) <= 1:
                        return True
                    return False

                link_ids = []

                (quad_tree, quad_table) = Quad.get_quad("A2")

                # - 1m 간격 좌표 배치
                stopLine = improve_points_density([points[0], points[-1]])

                for point in stopLine:

                    _link_ids = []
                    closest_p = get_closest_quad_point(point, quad_tree)

                    if closest_p == None:
                        break
                
                    if check_close(point, closest_p):
                        point2d = (closest_p[0], closest_p[1])
                        for link_id in quad_table[point2d]:
                            # - 이미 정지선이 등록된 A2 제외
                            if StopLine.get_stopLine(link_id=link_id) == None:
                                link = Shape.get_shape("A2", link_id)
                                if link != None:
                                    line = [stopLine[0], stopLine[-1]]
                                    if check_intersection_on_points(line, link.points):
                                        _link_ids.append(link_id)

                    link_ids += _link_ids

                # - 중복 제거
                link_ids = list(set(link_ids))

                return link_ids

        # --------------------------------------------------

        def execute(self):

            def save_stopLine():
                """
                정지선 맵
                """

                stopLine_datas = getattr(self.__class__, "stopLine_datas")

                map = Convert.create_map()

                for lane_id, stopLine in stopLine_datas.items():
                    line3d = Convert.convert_to_lineString3d(stopLine)
                    line3d.attributes["ID"]=  str(lane_id)
                    map.add(line3d)

                Convert.save_map("Shape_Generator", "stopLine.osm", map, sub_dir="Regulatory/StopLine")

            self.Generate_From_End().execute()
            self.Generate_From_Start().execute()
            self.Generate_From_Intersect().execute()

            save_stopLine()

    class TrafficLight_Generator():

        class Generate_TrafficLight():
            
            def extract_direction_datas(self):

                direction_datas = dict()

                light_datas = Shape.get_shape_datas("C1")
                counter = Process_Counter(len(light_datas))

                for light_id, light in light_datas.items():
                    direction = Regulatory_Generator.Direction.get_direction(light)
                    if direction != None:
                        direction_datas[light_id] = direction
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Light] Extract direction datas")
                counter.print_result("[Light] Extract direction datas")

                return direction_datas

            def create_refer_datas(self, direction_datas):

                light_datas = Shape.get_shape_datas("C1")
                counter = Process_Counter(len(light_datas))

                for light_id, light in light_datas.items():
                    # if Light.check_vehicle(light):
                    direction = direction_datas.get(light_id)
                    if direction != None:
                        origin = light.points[0]
                        code = light.Type
                        refer_width = Light.get_refer_width(code)
                        refer = convert_point_to_line(origin, direction[0], direction[-1], distance=refer_width)
                        Light.set_refer(light_id, refer)
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Light] Create refer datas")
                counter.print_result("[Light] Create refer datas")

            def create_bulb_datas(self, direction_datas):

                def get_origin(light, direction):
                    
                    light_origin = light.points[0]
                    bulb_gap = Light.get_bulb_gap()
                    gap_direction = rotate_seg(direction, deg=90)

                    bulb_origin = move_point(light_origin, None, start=gap_direction[0], end=gap_direction[-1], distance=bulb_gap)

                    return bulb_origin                

                class Bulb():

                    class Code_1():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(origin)
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval))

                            return bulb

                    class Code_2():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*1.5))
                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*1.5))

                            return bulb

                    class Code_3():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(origin)
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval))

                            return bulb

                    class Code_4():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(origin)
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval))

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(origin)
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval))

                            return bulb

                    class Code_5():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb

                    class Code_6():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*0.5))

                            return bulb

                    class Code_7():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb

                    class Code_8():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval))
                            bulb.append(origin)
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval))

                            return bulb

                    class Code_9():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(move_point(origin, None, start=light_dir[-1], end=light_dir[0], distance=bulb_interval*0.5))
                            bulb.append(move_point(origin, None, start=light_dir[0], end=light_dir[-1], distance=bulb_interval*0.5))

                            return bulb

                    class Code_10():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)

                            return bulb

                    class Code_11():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb

                    class Code_12():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb

                    class Code_13():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb
                            
                    class Code_14():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb
                            
                    class Code_15():

                        def create_bulb(self, origin, bulb_interval, light_dir):

                            bulb = []

                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)
                            bulb.append(origin)

                            return bulb

                    # --------------------------------------------------

                    @classmethod
                    def get_instance(cls, code):
                        if hasattr(cls, "Code_" + str(code)):
                            return getattr(cls, "Code_" + str(code))()

                def verify():

                    light_datas = Shape.get_shape_datas("C1")

                    for light_id in light_datas.keys():
                        if Light.get_bulb(light_id) == None:
                            log_print("Verify bulb : {0}".format(light_id))

                light_datas = Shape.get_shape_datas("C1")
                counter = Process_Counter(len(light_datas))

                for light_id, light in light_datas.items():
                    code = light.Type
                    direction = direction_datas.get(light_id)
                    if direction != None:
                        bulb_origin = get_origin(light, direction)
                        bulb_interval = Light.get_bulb_interval(code)
                        instance = Bulb.get_instance(code)
                        if instance != None:
                            bulb = instance.create_bulb(bulb_origin, bulb_interval, direction)
                            Light.set_bulb(light_id, bulb)
                        else:
                            counter.add(item="warn")
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Light] Create bulb datas")
                counter.print_result("[Light] Create bulb datas")

                verify()

            def rotate_light(self):
                """
                보행등, 자전거등에 해당하는 신호등 방향 조정            
                """

                def check_pedestrian(code):
                    """
                    보행등, 자전거등 여부 반환
                    """
                    if code in [11, 12, 13]:
                        return True
                    return False

                def calc_rotation(light, refer):
                    
                    (quad_tree, quad_table) = Quad.get_quad("A2")
                    origin = light.points[0]
                    closest_p = get_closest_quad_point(origin, quad_tree)

                    rotate_deg = 90
                    seg = rotate_seg(refer, deg=-90)
                    if not check_is_left(seg[0], seg[-1], closest_p):
                        rotate_deg = -90

                    return rotate_deg

                def rotate_refer(refer, rotate_deg):
                    refer = rotate_seg(refer, deg=rotate_deg)
                    return refer

                def rotate_bulbs(bulb, origin, rotate_deg):
                    bulb = rotate_seg(bulb, origin=origin, deg=rotate_deg)
                    return bulb

                _light_datas = dict()

                light_datas = Shape.get_shape_datas("C1")
                for light_id, light in light_datas.items():
                    if check_pedestrian(light.Type):
                        _light_datas[light_id] = light

                counter = Process_Counter(len(_light_datas))

                for light_id, light in _light_datas.items():
                    refer = Light.get_refer(light_id)
                    bulb = Light.get_bulb(light_id)
                    if None not in [refer, bulb]:
                        rotate_deg = calc_rotation(light, refer)
                        refer = rotate_refer(refer, rotate_deg)
                        bulb = rotate_bulbs(bulb, light.points[0], rotate_deg)
                        Light.set_refer(light_id, refer)
                        Light.set_bulb(light_id, bulb)
                        counter.add()
                    else:
                        counter.add(item="warn")
                    counter.print_sequence("[Light] Rotate light (Pedestrian)   ")
                counter.print_result("[Light] Rotate light (Pedestrian) ")

            # --------------------------------------------------

            def create_trafficLight(self):

                def extract_refer_keys(light):
                    """
                    동일한 A2 연관 C1 추출
                    - 1) A2 ID 동일 
                    - 2) 주행등
                    """

                    refer_keys = []

                    # 1) 대상 A2 ID 가 동일한 경우
                    lights = Light.get_lights(link_id=light.LinkID)
                    # 2) 주행등인 경우
                    lights = [x for x in lights if Light.check_vehicle(x)]

                    refer_keys = [x.ID for x in lights]                    

                    return refer_keys

                def extract_stopLine_key(light):
                    
                    stopLine_key = None

                    end_datas = getattr(Regulatory_Generator.StopLine_Generator, "End_datas")
                    start_datas = getattr(Regulatory_Generator.StopLine_Generator, "Start_datas")
                    intersect_datas = getattr(Regulatory_Generator.StopLine_Generator, "Intersect_datas")

                    link_id = light.LinkID

                    if Shape.check_shape("A2", link_id):
                        for datas in [end_datas, start_datas, intersect_datas]:
                            stopLine_key = datas.get(link_id)
                            if stopLine_key != None:
                                break

                    return stopLine_key

                def extract_lanelet_keys(light):

                    lanelet_keys = []

                    instance = Table.get_instance("C1", "A2", light.ID)
                    if instance != None:
                        origin_keys = instance.link_ids

                        for origin_key in origin_keys:
                            child_ids = A2.get_child_ids(origin_key)
                            lanelet_keys += child_ids

                    return lanelet_keys          

                light_datas = Shape.get_shape_datas("C1")
                counter = Process_Counter(len(light_datas))

                record = defaultdict(lambda : False)
                key = 0

                for light_id, light in light_datas.items():
                    if Light.check_vehicle(light):
                        if not record[light_id]:
                            refer_keys = extract_refer_keys(light)
                            if len(refer_keys) > 0:
                                stopLine_key = extract_stopLine_key(light)
                                if stopLine_key != None:
                                    lanelet_keys = extract_lanelet_keys(light)
                                    Light.set_trafficLight(key, refer_keys, stopLine_key, lanelet_keys)
                                    key += 1
                                    counter.add(item="fix")
                                record.update({x : True for x in refer_keys})
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Light] Create trafficLight")
                counter.print_result("[Light] Create trafficLight")

            def execute(self):

                def save_refer():

                    map = Convert.create_map()
            
                    light_datas = Shape.get_shape_datas("C1")
                    for light_id in light_datas.keys():
                        refer = Light.get_refer(light_id)
                        bulb = Light.get_bulb(light_id)
                        if None not in [refer, bulb]:
                            refer3d = Convert.convert_to_lineString3d(refer)
                            refer3d.attributes["ID"] = str(light_id)
                            bulbs3d = Convert.convert_to_lineString3d(bulb)
                            bulbs3d.attributes["ID"] = str(light_id)
                            map.add(refer3d)
                            map.add(bulbs3d)
                    
                    Convert.save_map("Shape_Generator", "refer.osm", map, sub_dir="Regulatory/Light")

                def save_trafficLight():

                    keys = Shape_Interface.Light.get_keys()

                    for key in keys:
                        map = Convert.create_map()
    
                        (refer_keys, stopLine_key, lanelet_keys) = Light.get_trafficLight(key)
                
                        if len(lanelet_keys) < 1:
                            continue

                        for refer_key in refer_keys:
                            refer = Light.get_refer(refer_key)
                            refer3d = Convert.convert_to_lineString3d(refer)
                            refer3d.attributes["C1"] = refer_key
                            map.add(refer3d)

                        stopLine = StopLine.get_stopLine(stopLine_key=stopLine_key)
                        stopLine3d = Convert.convert_to_lineString3d(stopLine)
                        map.add(stopLine3d)

                        for lanelet_key in lanelet_keys:
                            link = Shape.get_post("A2", lanelet_key)
                            line3d = Convert.convert_to_lineString3d(link.points)
                            line3d.attributes["A2"] = link.ID
                            
                            if Shape.get_shape("C1", refer_keys[0]).LinkID == lanelet_key:
                                line3d.attributes["Main"] = ""

                            map.add(line3d)

                        Convert.save_map("Shape_Generator", "{0}.osm".format(key), map, sub_dir="Regulatory/Light/trafficLight")

                def save_undone():

                    def save_vehicle(record):
                        
                        map = Convert.create_map()
                        for light_id, light in Shape.get_shape_datas("C1").items():
                            if not record[light_id]:
                                if Light.check_vehicle(light):
                                    refer = Light.get_refer(light_id)
                                    if refer != None:
                                        refer3d = Convert.convert_to_lineString3d(refer)
                                        refer3d.attributes["C1"] = light_id
                                        refer3d.attributes["A2"] = str(light.LinkID)
                                        map.add(refer3d)
                                        if Shape.check_shape("A2", light.LinkID):
                                            link = Shape.get_shape("A2", light.LinkID)
                                            line3d = Convert.convert_to_lineString3d(link.points)
                                            line3d.attributes["A2"] = str(link.ID)
                                            map.add(line3d)
                        Convert.save_map("Shape_Generator", "undone(vehicle).osm", map, sub_dir="Regulatory/Light")

                    def save_pedestrian(record):

                        map = Convert.create_map()
                        for light_id, light in Shape.get_shape_datas("C1").items():
                            if not record[light_id]:
                                if not Light.check_vehicle(light):
                                    refer = Light.get_refer(light_id)
                                    if refer != None:
                                        refer3d = Convert.convert_to_lineString3d(refer)
                                        refer3d.attributes["C1"] = light_id
                                        refer3d.attributes["A2"] = str(light.LinkID)
                                        map.add(refer3d)
                                        if Shape.check_shape("A2", light.LinkID):
                                            link = Shape.get_shape("A2", light.LinkID)
                                            line3d = Convert.convert_to_lineString3d(link.points)
                                            line3d.attributes["A2"] = str(link.ID)
                                            map.add(line3d)
                        Convert.save_map("Shape_Generator", "undone(pedestrian).osm", map, sub_dir="Regulatory/Light")

                    record = defaultdict(lambda : False)
                    for key in Shape_Interface.Light.get_keys():
                        refer_keys = Light.get_trafficLight(key)[0]
                        for refer_key in refer_keys:
                            record[refer_key] = True

                    save_vehicle(record)
                    save_pedestrian(record)

                direction_datas = self.extract_direction_datas()
                self.create_refer_datas(direction_datas)
                self.create_bulb_datas(direction_datas)
                self.rotate_light()
                self.create_trafficLight()

                save_refer()
                save_trafficLight()
                save_undone()

        # --------------------------------------------------

        def execute(self):
            self.Generate_TrafficLight().execute()

    class TrafficSign_Generator():

        class Generate_TrafficSign():

            def extract_direction_datas(self):

                direction_datas = dict()

                sign_datas = Shape.get_shape_datas("B1")
                counter = Process_Counter(len(sign_datas))

                for sign_id, sign in sign_datas.items():
                    direction = Regulatory_Generator.Direction.get_direction(sign)
                    if direction != None:
                        direction_datas[sign_id] = direction
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Sign] Extract direction datas")
                counter.print_result("[Sign] Extract direction datas")

                return direction_datas

            def create_refer_datas(self, direction_datas):
                
                sign_datas = Shape.get_shape_datas("B1")
                counter = Process_Counter(len(sign_datas))

                for sign_id, sign in sign_datas.items():
                    direction = direction_datas.get(sign_id)
                    if direction != None:
                        origin = sign.points[0]
                        refer_width = Sign.get_refer_width(sign_id)
                        refer = convert_point_to_line(origin, direction[0], direction[-1], distance=refer_width)
                        Sign.set_refer(sign_id, refer)
                    else:
                        counter.add(item="warn")
                    counter.add()
                    counter.print_sequence("[Sign] Create refer datas")
                counter.print_result("[Sign] Create refer datas")

            def create_trafficSign(self):

                def get_refer_keys(sign):
                    
                    refer_keys = []
                    
                    signs = Sign.get_signs(sign.LinkID)
                    refer_keys = [x.ID for x in signs]

                    _refer_keys = []

                    for refer_key in refer_keys:
                        refer = Sign.get_refer(sign_id)
                        if refer != None:
                            _refer_keys.append(refer_key)

                    refer_keys = _refer_keys

                    return refer_keys                    

                def get_stopLine_keys(refer_keys):
                    
                    stopLine_keys = []

                    links = [Sign.get_link(x) for x in refer_keys]

                    for link in links:
                        stopLine_key = StopLine.get_key(link.ID)
                        if stopLine_key != None:
                            if stopLine_key not in stopLine_keys:
                                stopLine_keys.append(stopLine_key)

                    # if len(stopLine_keys) < 1:
                    #     stopLine_keys = [None]

                    return stopLine_keys

                def get_lanelet_keys(sign):
                    
                    lanelet_keys = []

                    ref_lane = sign.Ref_Lane

                    origin_link = Sign.get_link(sign.ID)
                    child_ids = A2.get_child_ids(origin_link.ID)
                    
                    for child_id in child_ids:
                        child_neighbors = A2.get_neighbors(child_id)
                        for neighbor in child_neighbors:
                            if neighbor.LaneNo <= ref_lane:
                                lanelet_keys.append(neighbor.ID)

                    return lanelet_keys

                def group_signs(refer_keys):
                
                    groups = defaultdict(lambda : [])

                    for refer_key in refer_keys:
                        code = Sign.get_code(refer_key)
                        groups[code].append(refer_key)

                    return groups

                sign_datas = Shape.get_shape_datas("B1")
                counter = Process_Counter(len(sign_datas))

                key = 0
                record = defaultdict(lambda : False)

                for sign_id, sign in sign_datas.items():
                    if not record[sign_id]:
                        if Sign.get_link(sign_id) != None:
                            refer_keys = []
                            for refer_key in get_refer_keys(sign):
                                refer = Sign.get_refer(refer_key)
                                if refer != None:
                                    refer_keys.append(refer_key)
                            if len(refer_keys) > 0:
                                stopLine_keys = get_stopLine_keys(refer_keys)
                                lanelet_keys = get_lanelet_keys(sign)
                                groups = group_signs(refer_keys)
                                for stopLine_key in stopLine_keys:
                                    for code, _refer_keys in groups.items():
                                        Sign.set_trafficSign(key, code, _refer_keys, stopLine_key, lanelet_keys)
                                        counter.add(item="fix")
                                        key = key + 1
                            record.update({x.ID : True for x in Sign.get_signs(sign.LinkID)})
                    counter.add()
                    counter.print_sequence("[Sign] Create trafficSign")
                counter.print_result("[Sign] Create trafficSign")
    
            def create_trafficLight(self):

                def extract_refer_keys(sign):
                    """
                    동일한 A2 연관 B1 추출
                    - 1) A2 ID 동일 
                    """

                    refer_keys = []

                    # 1) 대상 A2 ID 가 동일한 경우
                    signs = Sign.get_signs(link_id=sign.LinkID)

                    refer_keys = [x.ID for x in signs]                    

                    return refer_keys

                def extract_stopLine_key(sign):
                    
                    stopLine_key = None

                    end_datas = getattr(Regulatory_Generator.StopLine_Generator, "End_datas")
                    start_datas = getattr(Regulatory_Generator.StopLine_Generator, "Start_datas")
                    intersect_datas = getattr(Regulatory_Generator.StopLine_Generator, "Intersect_datas")

                    link_id = sign.LinkID

                    if Shape.check_shape("A2", link_id):
                        for datas in [end_datas, start_datas, intersect_datas]:
                            stopLine_key = datas.get(link_id)
                            if stopLine_key != None:
                                break

                    return stopLine_key

                def extract_lanelet_keys(sign):

                    lanelet_keys = []

                    link_id = sign.LinkID

                    origin_keys = []

                    neighbors = A2.get_neighbors(link_id, is_post=False)
                    for neighbor in neighbors:
                        origin_keys.append(neighbor.ID)

                    for origin_key in origin_keys:
                        child_ids = A2.get_child_ids(origin_key)
                        lanelet_keys += child_ids

                    return lanelet_keys

                sign_datas = Shape.get_shape_datas("B1")
                counter = Process_Counter(len(sign_datas))

                record = defaultdict(lambda : False)
                key = 0

                for sign_id, sign in sign_datas.items():
                    if not record[sign_id]:
                        refer_keys = extract_refer_keys(sign)
                        if len(refer_keys) > 0:
                            stopLine_key = extract_stopLine_key(sign)
                            if stopLine_key != None:
                                lanelet_keys = extract_lanelet_keys(sign)
                                Sign.set_trafficSign(key, refer_keys, stopLine_key, lanelet_keys)
                                key += 1
                                counter.add(item="fix")
                            record.update({x : True for x in refer_keys})
                    counter.add()
                    counter.print_sequence("[Sign] Create trafficSign")
                counter.print_result("[Sign] Create trafficSign")

            # --------------------------------------------------

            def execute(self):

                def save_refer():

                    map = Convert.create_map()
            
                    sign_datas = Shape.get_shape_datas("B1")
                    for sign_id in sign_datas.keys():
                        refer = Sign.get_refer(sign_id)
                        if refer != None:
                            refer3d = Convert.convert_to_lineString3d(refer)
                            map.add(refer3d)

                    Convert.save_map("Shape_Generator", "refer.osm", map, sub_dir="Regulatory/TrafficSign")

                def save_trafficSign():

                    keys = Shape_Interface.Sign.get_keys()
                    for key in keys:

                        map = Convert.create_map()
    
                        (code, refer_keys, stopLine_key, lanelet_keys) = Sign.get_trafficSign(key)
                        for refer_key in refer_keys:
                            refer = Sign.get_refer(refer_key)
                            refer3d = Convert.convert_to_lineString3d(refer)
                            refer3d.attributes["B1"] = refer_key
                            map.add(refer3d)

                        stopLine = StopLine.get_stopLine(stopLine_key=stopLine_key)
                        stopLine3d = Convert.convert_to_lineString3d(stopLine)
                        map.add(stopLine3d)

                        for lanelet_key in lanelet_keys:
                            line = Shape.get_post("A2", lanelet_key).points
                            line3d = Convert.convert_to_lineString3d(line)
                            line3d.attributes["A2"] = lanelet_key
                            if Shape.get_shape("B1", refer_keys[0]).LinkID == lanelet_key:
                                line3d.attributes["Main"] = ""
                            map.add(line3d)

                        Convert.save_map("Shape_Generator", "{0}.osm".format(key), map, sub_dir="Regulatory/Sign/trafficSign")

                direction_datas = self.extract_direction_datas()
                self.create_refer_datas(direction_datas)
                self.create_trafficSign()

                save_refer()
                save_trafficSign()

        # --------------------------------------------------

        def execute(self):
            self.Generate_TrafficSign().execute()

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.StopLine_Generator().execute()
        cls.TrafficLight_Generator().execute()
        # cls.TrafficSign_Generator().execute()


class Shape_Generator(Module):
    
    def do_process(cls, *args, **kwargs):
        Road_Generator.execute()
        Crosswalk_Generator.execute()
        Regulatory_Generator.execute()
        Shape.add_interface()

        return True


