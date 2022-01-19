#!/usr/bin/python
# -*- coding: utf-8 -*-

from six import add_metaclass
from abc import (
    ABCMeta,
    abstractmethod,
)
from collections import defaultdict
from functools import reduce
from copy import deepcopy
from numpy import (
    average,
    quantile,
)

from scripts.shape.shape_module import Module
from scripts.shape.shape_data import (
    Shape,
    A2,
    Convert,
)
from scripts.functions.coordinate_functions import (
    calc_curve_diff,
    check_point_on_line,
    get_closest_point,
    get_closest_segment,
    get_inner_segment,
    get_intersection_on_points,
    get_point_on_points,
    calc_distance,
    calc_distance_from_line,
    get_ortho_line,
    improve_points_density,
    parse_points,
    get_mid,
    calc_length,
    get_point_on_points,
    move_line,
    calc_parallel,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
    interrupt_print,
)


class A2_Parser():
    
    class Check_Multiple():
        """
        이웃 A2 간 1:N 관계 검출
        """
        
        def execute(self):

            def save(multiple_datas):

                if len(multiple_datas) < 1:
                    return

                map = Convert.create_map()
                for link_id, (left_ids, right_ids) in multiple_datas.items():
                    link = Shape.get_shape("A2", link_id)
                    line3d = Convert.convert_to_lineString3d(link.points)
                    map.add(line3d)
                    for neighbor_ids in [left_ids, right_ids]:
                        for neighbor_id in neighbor_ids:
                            neighbor_link = Shape.get_shape("A2", neighbor_id)
                            relation_line = [
                                get_point_on_points(link.points, None, division=2),
                                get_point_on_points(neighbor_link.points, None, division=2)
                            ]
                            relation_line3d = Convert.convert_to_lineString3d(relation_line)
                            map.add(relation_line3d)
                Convert.save_map("Shape_Parser", "Check_Multiple.osm", map)

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            # 1. 이웃관계 count 
            relation_datas = defaultdict(lambda : ([], []))
            for link_id, link in link_datas.items():
                # - 좌/우 이웃 추출
                for side_index, side_link in enumerate([Shape.get_shape("A2", link.L_LinkID), Shape.get_shape("A2", link.R_LinkID)]):
                    # - 이웃 존재 시
                    if side_link != None:
                        # - 이웃 count + 1
                        relation_datas[side_link.ID][1-side_index].append(link_id)
                counter.add()
                counter.print_sequence("[Multiple] Check multiple")

            # 2. Multiple 관계 추출
            # - Multiple : 이웃 count > 1 
            multiple_datas = {x[0] : x[1] for x in relation_datas.items() if any([len(x[1][index]) > 1 for index in [0, 1]])}
            counter.add(item="warn", count=len(multiple_datas))
            counter.print_result("[Multiple] Check multiple")

            save(multiple_datas)

            # - Multiple 관계 존재 시 중단
            if len(multiple_datas) > 0:
                interrupt_print("Stop parsing => Multiple A2 relation detected ({0})".format(len(multiple_datas)))

    @add_metaclass(ABCMeta)
    class Parse_Process():
        
        options = {
            "Parse_Unbalance" : "unbalance",
            "Parse_Unstable" : "unstable",
            "Parse_Length" : "length",
            "Parse_Lane" : "lane",
        }

        road_max = 6.0
        road_min = 2.0

        def get_option(self):
            return self.__class__.options.get(self.__class__.__name__)

        def extract_neighbor_datas(self):

            def check_count(neighbors):

                neighbor_count = {
                    "unbalance" : 2,
                    "unstable" : 2,
                    "length" : 1,
                    "lane" : 1,
                }.get(self.get_option())

                if len(neighbors) < neighbor_count:
                    return False
                
                return True

            def check_length(neighbors, length_limit=1.0):
                """
                Lanelet 최소길이 = 1.0m
                """

                min_link = A2.select_min(neighbors)
                
                if min_link.Length <= length_limit:
                    return False

                return True

            def verify(neighbor_datas):
                
                record = defaultdict(lambda : 0)

                for neighbors in neighbor_datas.values():
                    for link in neighbors:
                        record[link.ID] += 1

                errors = 0
                for link_id, count in record.items():
                    if count > 1:
                        errors += 1
                
                warning_print("Verify : {0}".format(errors))

            neighbor_datas = dict()

            link_datas = Shape.get_post_datas("A2")
            counter = Process_Counter(len(link_datas))

            record = defaultdict(lambda : False)

            for link_id, link in link_datas.items():
                if not record[link_id]: 
                    neighbors = A2.get_neighbors(link_id)
                    if check_count(neighbors):
                        if check_length(neighbors):
                            neighbor_datas[len(neighbor_datas)] = neighbors                    
                            counter.add(item="fix")
                    record.update({x.ID : True for x in neighbors})
                counter.add()
                counter.print_sequence("[{0}] Extract neighbor datas".format(self.get_option()))
            counter.print_result("[{0}] Extract neighbor datas".format(self.get_option()))

            # verify(neighbor_datas)

            return neighbor_datas

        def parse_neighbor_datas(self, neighbor_datas, material_datas):
            
            def create_nodes(parent_node, data_list):
                
                new_nodes = []
            
                points = [x["points"][-1] for x in data_list[:-1]]

                for point in points:
                    new_node = (deepcopy(parent_node)
                        .replace(ID=Shape.generate_key("A1"))
                        .replace(points=[point])
                        )
                    new_nodes.append(new_node)
                
                return new_nodes

            def create_links(parent_link, new_nodes, materials, data_list):
                
                new_links = []

                # 1. 신규 A2 목록 생성
                for index, data in enumerate(data_list):

                    # 1) 신규 A2 정보 생성
                    # - 동일한 분할정보에 의해 분할된 A2 는 동일한 ID 순번을 가진다.
                    # - 마지막 이전 분할정보에 해당하는 경우 ["index"] 값을 사용
                    # - 마지막 이후 (1개) 분할정보에 해당하는 경우 분할정보 개수 + 1 에 해당하는 순번 사용 (분할정보는 순서가 없기 때문)
                    parse_index = data["index"] if data["is_prev"] else len(materials)
                    new_ID = parent_link.ID + "_{0}".format(parse_index)
                    new_L_LinkID = parent_link.L_LinkID + "_{0}".format(parse_index) if parent_link.L_LinkID != -1 else -1
                    new_R_LinkID = parent_link.R_LinkID + "_{0}".format(parse_index) if parent_link.R_LinkID != -1 else -1
                    new_FromNodeID = new_nodes[index].ID
                    new_ToNodeID = new_nodes[index+1].ID
                    new_points = data["points"]

                    # 2) 신규 A2 생성
                    new_link = (deepcopy(parent_link)
                        .replace(ID=new_ID)
                        .replace(L_LinkID=new_L_LinkID)
                        .replace(R_LinkID=new_R_LinkID)
                        .replace(FromNodeID=new_FromNodeID)
                        .replace(ToNodeID=new_ToNodeID)
                        .replace(points=new_points)
                        .replace(Length=calc_length(new_points))
                        )

                    # 3) 분할 과정에 따른 이웃관계 갱신
                    self._set_relation(parent_link, new_link, materials, data_list, index)

                    new_links.append(new_link)

                if self.get_option() != "unbalance":
                    if len(new_links) < len(materials) + 1:
                        warning_print("Parse error = {0} - {1} / {2}".format(parent_link.ID, len(materials), len(new_links)))

                return new_links

            new_node_datas = dict()
            new_link_datas = dict()

            counter = Process_Counter(len([x for x in neighbor_datas.values() if len(x) > 0]))

            # 1. 예외 관계 제거
            # - 분할 대상 / 분할정보 생성 실패
            for key, neighbors in neighbor_datas.items():
                materials = material_datas[key]
                _materials = [mat for mat in materials if mat.get("parse_line") == None]
                for mat in _materials:
                    main = next(x for x in neighbors if x.ID == mat["main"])
                    sub = next(x for x in neighbors if x.ID == mat["sub"])
                    for side_index in [0, 1]:
                        if getattr(main, ["L_LinkID", "R_LinkID"][side_index]) == sub.ID:
                            main.replace(**{["L_LinkID", "R_LinkID"][side_index] : -1})
                            sub.replace(**{["L_LinkID", "R_LinkID"][1-side_index] : -1})
                            counter.add(item="fix")
                            break
                counter.add()
                counter.print_sequence("[{0}] Break neighbor datas".format(self.get_option()))
            counter.print_result("[{0}] Break neighbor datas".format(self.get_option()))

            counter = Process_Counter(len([x for x in neighbor_datas.values() if len(x) > 0]))

            # 2. 분할
            # - 분할 대상 / 분할정보 생성 성공
            for key, neighbors in neighbor_datas.items():
                materials = material_datas[key]
                _materials = [mat for mat in materials if mat.get("parse_line") != None]
                if len(_materials) > 0:
                    parse_lines = [x["parse_line"] for x in _materials]
                    for link in neighbors:
                        data_list = parse_points(link.points, parse_lines)

                        _new_nodes = create_nodes(Shape.get_post("A1", link.FromNodeID), data_list)
                        _new_links = create_links(
                            link, 
                            [Shape.get_post("A1", link.FromNodeID)] + _new_nodes + [Shape.get_post("A1", link.ToNodeID)], 
                            _materials,
                            data_list
                            )  

                        new_node_datas[link.ID] = _new_nodes
                        new_link_datas[link.ID] = _new_links
                        counter.add(item="fix", count=len(_new_links)-1)
                counter.add()
                counter.print_sequence("[{0}] Parse neighbor datas".format(self.get_option()))
            counter.print_result("[{0}] Parse neighbor datas".format(self.get_option()))

            return new_node_datas, new_link_datas

        def extract_pairs(self, neighbors):

            pairs = []

            record = defaultdict(lambda : False)

            for link in neighbors:
                for side_link in [Shape.get_post("A2", getattr(link, item)) for item in ["L_LinkID", "R_LinkID"]]:
                    if side_link != None:
                        (main, sub) = A2.classify_pair(link, side_link)
                        key = "{0}/{1}".format(main.ID, sub.ID)
                        if not record[key]:     
                            pairs.append((main, sub))
                            record[key] = True

            return pairs

        def update_shape(self, new_node_datas, new_link_datas):
            
            def verify():
                
                for parent_id, child_ids in Shape.get_shape_datas("A2_RECORD").items():
                    if Shape.get_post("A2", parent_id) != None:
                        warning_print("Verify : {0} - {1}".format(parent_id, child_ids))

            # 1. 신규 A1 추가
            for new_nodes in new_node_datas.values():
                for new_node in new_nodes:
                    Shape.set_post("A1", new_node)

            # 2. 신규 A2 추가 (자식) / 기존 A2 제거 (부모)
            for parent_id, child_links in new_link_datas.items():
                for child_link in child_links:
                    # - 목록 갱신
                    Shape.del_post("A2", parent_id)
                    Shape.set_post("A2", child_link)
                    # - 부모-자식 관계 기록
                    A2.record_parse(parent_id, child_link.ID)

            # 3. 신규 A2 이웃관계 정리
            # - A2 분할 시 이웃 내 분할개수가 차이나는 경우 존재하지 않는 신규 A2 를 이웃으로 참조할 수 있음.
            for parent_id, child_links in new_link_datas.items():
                for child_link in child_links:
                    for item in ["L_LinkID", "R_LinkID"]:
                        side_id = getattr(child_link, item)
                        if side_id != -1:
                            if Shape.get_post("A2", side_id) == None:
                                child_link.replace(**{item : -1})

            # verify()            

        # --------------------------------------------------

        @abstractmethod
        def extract_source_datas(self, neighbor_datas):
            pass

        @abstractmethod
        def create_material_datas(self, source_datas):
            pass

        # --------------------------------------------------

        def except_material_datas(self, neighbor_datas, material_datas):

            counter = Process_Counter(len(material_datas))

            for key, materials in material_datas.items():
                neighbors = neighbor_datas[key]
                
                extra_materials = [mat for mat in materials if mat.get("parse_line") == None]
                _materials = [mat for mat in materials if mat.get("parse_line") != None]
                _materials = self._except_materials(neighbors, _materials)
                material_datas[key] = _materials + extra_materials

                counter.add(item="fix", count=len(materials) - len(_materials))
                counter.add()
                counter.print_sequence("[{0}] Except material datas".format(self.get_option()))
            counter.print_result("[{0}] Except material datas".format(self.get_option()))

            return material_datas

        def _except_materials(self, neighbors, materials):
            """
            기본 제외 규칙 (Unbalance 제외)
            1) 교차로 제외
            2) 미교차 분할정보 제외
            3) 좌/우 이웃 간 분할지점 거리가 기준(1.5m) 이하인 경우 제외
            4) 분할지점 간 거리가 기준(2.0m) 이하인 경우 제외
            """

            def except_intersection(neighbors, materials):
                """
                분할 대상 중 교차로 주행로가 있다면 분할 제외
                """

                for link in neighbors:
                    if A2.check_intersection(link.ID):
                        materials = []
                        break

                return materials

            def except_not_intersect(neighbors, materials):

                for link in neighbors:
                    _list = []
                    for mat in materials:
                        intersect_p = get_intersection_on_points(mat["parse_line"], link.points)
                        if intersect_p == None:
                            _list.append(mat)
                    materials = [mat for mat in materials if mat not in _list]                    

                return materials

            def except_close_neighbor(neighbors, materials, min_dist=1.5):

                _materials = []

                indices = dict()

                for index, mat in enumerate(materials):
                    
                    close_check = False

                    # - 분할선과 A2 간 교차점 추출
                    intersect_points = []
                    for link in neighbors:
                        intersect_p = get_intersection_on_points(mat["parse_line"], link.points)
                        if intersect_p != None:
                            intersect_points.append(intersect_p)

                    # - 교차점 쌍 사이의 거리가 기준 이하인 경우 제외
                    for i in range(len(intersect_points) - 1):
                        for j in range(i + 1, len(intersect_points)):
                            A = intersect_points[i]
                            B = intersect_points[j]
                            dist = calc_distance(A, B)
                            if dist < min_dist:
                                indices[index] = mat
                                close_check = True
                                break
                        if close_check:
                            break

                for index, mat in enumerate(materials):
                    if indices.get(index) == None:
                        _materials.append(mat)

                return _materials

            def except_close(neighbors, materials, close_dist=2.0):

                # 1. 시작/종료 분할정보 제거
                # - 시작 좌표 / 종료 좌표와 1.0 m 이하 간격 시 제거
                _list = []
                for link in neighbors:
                    for index in [0, -1]:
                        point = link.points[index]
                        for mat in materials:
                            intersect_p = get_intersection_on_points(mat["parse_line"], link.points)
                            if intersect_p != None:
                                if calc_distance(point, intersect_p) <= close_dist:
                                    if mat not in _list:
                                        _list.append(mat)
                materials = [mat for mat in materials if mat not in _list]

                # 2. 분할정보 간 간격 이하 제거
                _materials = materials[:1]
                for index in range(1, len(materials)):
                    
                    prev_mat = _materials[-1]
                    curr_mat = materials[index]

                    if calc_distance(prev_mat["parse_line"][1], curr_mat["parse_line"][1]) > close_dist:
                        _materials.append(curr_mat)

                return _materials

            materials = except_intersection(neighbors, materials)
            materials = except_not_intersect(neighbors, materials)
            materials = except_close_neighbor(neighbors, materials)
            materials = except_close(neighbors, materials)

            return materials

        def except_cross_type(self, neighbors, materials):
            """
            A2 에 등록된 B2 목록 중 좌점혼선 / 우점혼선에 해당 시 분할정보 제외                
            """

            def check_cross_type(lane):
                if lane.Type % 10 in [3, 4]:
                    return True
                return False

            for link in neighbors:
                (left_lanes, right_lanes) = A2.get_lanes(link.ID)
                for lane in left_lanes + right_lanes:
                    if check_cross_type(lane):
                        materials = []
                        break                            
                if len(materials) < 1:
                    break

            return materials

        def except_merge(self, neighbors, materials, max_diff=10.0):
            """
            * parse_unstable 이후 분할 과정에 반드시 포함
            - 병합 + 수렴영역 해당 시 제외
                - 수렴영역 : 평행하지 않고 좌/우 이웃 간 거리가 좁아지는 영역
                * 수렴영역 내 분할 시 전/후 centerLine 이 왜곡되는 문제가 발생
            """

            def check_lane(link_id):
                """
                A2 의 좌/우 B2 중 좌점혼선, 우점혼선에 해당하는 B2 의 존재여부 반환
                """

                left_lanes, right_lanes = A2.get_lanes(link_id)

                for lane in left_lanes + right_lanes:
                    if lane.Type % 10 in [3, 4]:
                        return True

                return False

            indices = []

            pairs = self.extract_pairs(neighbors)

            for (main, sub) in pairs:
                for side_index in [0, 1]:
                    if A2.check_merge(sub.ID, side_index):
                        #* 좌점혼선 / 우점혼선 차선에 해당하는 경우 분할정보 전체 제거
                        if check_lane(sub.ID) or check_lane(main.ID):
                            return []
                        for index, mat in enumerate(materials):
                            if index not in indices:

                                parse_line = mat["parse_line"]

                                # - 분할선과 교차하는 Sub 선분 추출 
                                sub_point = get_intersection_on_points(parse_line, sub.points)
                                A = get_inner_segment(sub_point, sub.points)

                                # - 분할선과 교차하는 Main 선분 추출
                                main_point = get_intersection_on_points(parse_line, main.points)
                                B = get_inner_segment(main_point, main.points)

                                # - Sub / Main 선분간 각도차가 기준(max_diff) 초과 시 분할정보 제외 (= 수렴영역 판정)
                                if calc_curve_diff(A, compare_points=B) > max_diff:
                                    indices.append(index)

            materials = [mat for mat in materials if materials.index(mat) not in indices]

            return materials

        def except_branch(self, neighbors, materials, max_diff=10.0):
            """
            * parse_unstable 이후 분할 과정에 반드시 포함
            - 분기 + 발산영역 해당 시 제외
                - 발산영역 : 평행하지 않고 좌/우 이웃 간 거리가 넓어지는 영역
                * 발산영역 내 분할 시 전/후 centerLine 이 왜곡되는 문제가 발생
            """

            def check_lane(link_id):
                """
                A2 의 좌/우 B2 중 좌점혼선, 우점혼선에 해당하는 B2 의 존재여부 반환
                """

                left_lanes, right_lanes = A2.get_lanes(link_id)

                for lane in left_lanes + right_lanes:
                    if lane.Type % 10 in [3, 4]:
                        return True

                return False

            indices = []

            pairs = self.extract_pairs(neighbors)
            for (main, sub) in pairs:
                for side_index in [0, 1]:
                    if A2.check_branch(sub.ID, side_index):
                        #* 좌점혼선 / 우점혼선 차선에 해당하는 경우 분할정보 전체 제거
                        if check_lane(sub.ID) or check_lane(main.ID):
                            return []
                        for index, mat in enumerate(materials):
                            if index not in indices:
                                
                                parse_line = mat["parse_line"]
                                
                                # - 분할선과 교차하는 Sub 선분 추출 
                                sub_point = get_intersection_on_points(parse_line, sub.points)
                                A = get_inner_segment(sub_point, sub.points)

                                # - 분할선과 교차하는 Main 선분 추출
                                main_point = get_intersection_on_points(parse_line, main.points)
                                B = get_inner_segment(main_point, main.points)
                                                                
                                # - Sub / Main 선분간 각도차가 기준(max_diff) 초과 시 분할정보 제외 (= 발산영역 판정)
                                if calc_curve_diff(A, compare_points=B) > max_diff:
                                    indices.append(index)

            materials = [mat for mat in materials if materials.index(mat) not in indices]

            return materials

        @abstractmethod
        def _set_relation(self, parent_link, new_link, materials, data_list, data_index):
            pass

        # --------------------------------------------------

        @abstractmethod
        def _save_source(self, source_datas):
            pass

        @abstractmethod
        def _save_material(self, material_datas):
            pass

        def _save_new_links(self, new_link_datas):
            
            map = Convert.create_map()

            for new_links in new_link_datas.values():
                for new_link in new_links:
                    line3d = Convert.convert_to_lineString3d(new_link.points, link_id=new_link.ID)
                    line3d.attributes["ID"] = str(new_link.ID)
                    line3d.attributes["FromNode"] = str(new_link.FromNodeID)
                    line3d.attributes["ToNode"] = str(new_link.ToNodeID)
                    map.add(line3d)
                    r_link = Shape.get_post("A2", new_link.R_LinkID)
                    if r_link != None:
                        map.add(Convert.convert_to_lineString3d([
                            get_point_on_points(new_link.points, None, division=2),
                            get_point_on_points(r_link.points, None, division=2)
                        ]))

            Convert.save_map("Shape_Parser", "new_link.osm", map, sub_dir="{0}".format(self.get_option()))

        def execute(self):

            neighbor_datas = self.extract_neighbor_datas()

            source_datas = self.extract_source_datas(neighbor_datas)
            self._save_source(source_datas)

            material_datas = self.create_material_datas(source_datas)
            material_datas = self.except_material_datas(neighbor_datas, material_datas)
            self._save_material(material_datas)

            new_node_datas, new_link_datas = self.parse_neighbor_datas(neighbor_datas, material_datas)

            self.update_shape(new_node_datas, new_link_datas)
            self._save_new_links(new_link_datas)

    class Parse_Unbalance(Parse_Process):
        
        def extract_source_datas(self, neighbor_datas):

            def check_unbalance(main, sub):
                for index in [0, -1]:
                    if self._check_unbalance(main, sub, index):
                        return True
                return False

            source_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(neighbor_datas))

            for key, neighbors in neighbor_datas.items():
                pairs = self.extract_pairs(neighbors)
                for (main, sub) in pairs:
                    if check_unbalance(main, sub):
                        source_datas[key].append((main, sub))
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[{0}] Extract source datas".format(self.get_option()))
            counter.print_result("[{0}] Extract source datas".format(self.get_option()))

            return source_datas

        def create_material_datas(self, source_datas):

            def create_materials(main, sub):

                def check_length(main, sub):

                    parallel_length = calc_parallel(main.points, sub.points, error_degree=2.5)

                    if parallel_length < 10.0:
                        return False
                    return True

                def extract_parse_point(main, sub, index):
                    main_dist = calc_distance(main.points[index], get_closest_point(main.points[index], sub.points))
                    sub_dist = calc_distance(sub.points[index], get_closest_point(sub.points[index], main.points))
                    parse_point = main.points[index] if main_dist <= sub_dist else sub.points[index]
                    return parse_point

                def extract_parse_point(main, sub, index):
                    
                    parse_point = None

                    main_points = main.points if index == 0 else main.points[::-1]
                    sub_points = sub.points

                    main_points = improve_points_density(main_points)
                    sub_points = improve_points_density(sub_points)

                    for main_p in main_points[1:-1]:
                        sub_p = get_closest_point(main_p, sub_points)
                        if calc_distance(main_p, sub_p) < self.__class__.road_max:
                            parse_point = main_p
                            break
                    
                    return parse_point

                materials = []

                for index in [0, -1]:
                    if self._check_unbalance(main, sub, index):
                        # 1) 기준길이 이하 : 이웃관계 제거
                        if not check_length(main, sub):
                            material = {
                                "main" : main.ID,
                                "sub" : sub.ID
                            }
                        # 2) 기준길이 이상 : 분할
                        else:
                            # parse_point = extract_parse_point(main, sub, index)
                            # direction = get_closest_segment(parse_point, main.points)
                            # parse_line = get_ortho_line(parse_point, start=direction[0], end=direction[-1])
                            # material = {
                            #     "parse_line" : parse_line,
                            # }
                            parse_point = extract_parse_point(main, sub, index)
                            if parse_point != None:
                                direction = get_closest_segment(parse_point, main.points)
                                parse_line = get_ortho_line(parse_point, start=direction[0], end=direction[-1])
                                material = {
                                    "parse_line" : parse_line,
                                }
                            
                        materials.append(material)
                
                return materials

            material_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(source_datas))

            for key, pairs in source_datas.items():
                for (main, sub) in pairs:
                    materials = create_materials(main, sub)
                    material_datas[key] += materials
                    counter.add(item="fix", count=len(materials))
                counter.add()
                counter.print_sequence("[{0}] Create material datas".format(self.get_option()))
            counter.print_result("[{0}] Create material datas".format(self.get_option()))

            return material_datas

        def _except_materials(self, neighbors, materials):
            """
            Unbalance 분할은 분할선이 교차하지 않아도 제외하지 않는다. 
            """
            
            return materials

        # --------------------------------------------------

        def _check_unbalance(self, main, sub, index):
            main_p = main.points[index]
            sub_p = sub.points[index]
            if calc_distance(main_p, sub_p) > self.__class__.road_max:
                direction = main.points[:2] if index == 0 else main.points[-2:][::-1]
                ortho_line = get_ortho_line(main_p, start=direction[0], end=direction[-1])
                if calc_distance_from_line(ortho_line, sub_p) > self.__class__.road_max:
                    return True
            return False

        def _set_relation(self, parent_link, new_link, materials, data_list, data_index):
            pass

        # --------------------------------------------------
        
        def _save_source(self, source_datas):

            map = Convert.create_map()
            
            for pairs in source_datas.values():
                for (main, sub) in pairs:
                    map.add(Convert.convert_to_lineString3d(main.points))
                    map.add(Convert.convert_to_lineString3d(sub.points))
                    map.add(Convert.convert_to_lineString3d([
                        get_point_on_points(main.points, None, division=2),
                        get_point_on_points(sub.points, None, division=2)
                    ]))

            Convert.save_map("Shape_Parser", "source.osm", map, sub_dir="{0}".format(self.get_option()))

        def _save_material(self, material_datas):

            map = Convert.create_map()

            for materials in material_datas.values():
                for material in materials:
                    if material.get("parse_line") == None:
                        main = Shape.get_post("A2", material["main"])
                        sub = Shape.get_post("A2", material["sub"])
                        map.add(Convert.convert_to_lineString3d(main.points))
                        map.add(Convert.convert_to_lineString3d(sub.points))
                        map.add(Convert.convert_to_lineString3d([
                            get_point_on_points(main.points, division=2),
                            get_point_on_points(sub.points, division=2),
                        ]))
                    else:
                        map.add(Convert.convert_to_lineString3d(material["parse_line"]))
            
            Convert.save_map("Shape_Parser", "material.osm", map, sub_dir="{0}".format(self.get_option()))

    class Parse_Unstable(Parse_Process):

        def extract_source_datas(self, neighbor_datas):
            
            source_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(neighbor_datas))

            for key, neighbors in neighbor_datas.items():
                pairs = self.extract_pairs(neighbors)
                for (main, sub) in pairs:
                    if self._check_unstable(main, sub):
                        source_datas[key].append((main, sub))
                counter.add(item="fix", count=len(source_datas[key]))
                counter.add()
                counter.print_sequence("[{0}] Extract source datas".format(self.get_option()))
            counter.print_result("[{0}] Extract source datas".format(self.get_option()))

            return source_datas

        def create_material_datas(self, source_datas):

            def create_materials(main, sub):

                def calc_standard(main, sub):

                    dist_list = []

                    for sub_p in sub.points:
                        main_seg = get_closest_segment(sub_p, main.points)
                        dist = calc_distance_from_line(main_seg, sub_p)
                        if (self.__class__.road_min <= dist <= self.__class__.road_max):
                            dist_list.append(dist)

                    if len(dist_list) > 0:
                        standard = average([dist for dist in dist_list if quantile(dist_list, 0.5) >= dist])
                        # standard = quantile(dist_list, .75)
                        # standard = average(dist_list)
                    else:
                        standard = None

                    return standard

                materials = []

                # 1. 평균 거리 / 최소 / 최대 추출
                standard = calc_standard(main, sub)
                
                # - 평균 거리 검출 실패 시 (모든 좌표간 거리가 미만 or 초과)
                if standard == None:
                    material = {
                        "main" : main.ID,
                        "sub" : sub.ID,
                    }
                    materials.append(material)
                # - 평균 거리 검출 성공 시
                else:
                    min_dist = standard * (1.0 - 0.2)
                    max_dist = standard * (1.0 + 1.0)

                    # - 검색된 범위
                    index = 0
                    range_index = -1

                    while index < len(sub.points):

                        sub_p = sub.points[index]
                        main_seg = get_closest_segment(sub_p, main.points)

                        curr_dist = calc_distance_from_line(main_seg, sub_p)
                        curr_stat = not (min_dist <= curr_dist <= max_dist)

                        if curr_stat:
                            direction = 1 if index == 0 else -1
                            checker = (lambda dist : dist >= standard) if curr_dist < min_dist else (lambda dist : dist <= max_dist)
                            # - 검색
                            indices = range(index, len(sub.points)) if direction == 1 else range(index, -1, -1)
                            for _index in indices:

                                if _index <= range_index:
                                    break

                                _sub_p = sub.points[_index]
                                _main_seg = get_closest_segment(_sub_p, main.points)
                                _curr_dist = calc_distance_from_line(_main_seg, _sub_p)
                                # - 분할지점 검색 or 검색범위 종료 시
                                if checker(_curr_dist) or _index == indices[-1]:
                                    parse_line = get_ortho_line(_sub_p, start=_main_seg[0], end=_main_seg[-1])
                                    material = {
                                        "parse_line" : parse_line,
                                        "main" : main.ID,
                                        "sub" : sub.ID,
                                        "prev_stat" : True if direction == 1 else False,
                                    }                               
                                    materials.append(material)
                                    # - 검색된 범위 갱신
                                    range_index = max([index, _index])
                                    break 

                        index = max([index+1, range_index])

                return materials

            material_datas = defaultdict(lambda : [])

            _source_datas = {x[0] : x[1] for x in source_datas.items() if len(x[1]) > 0}
            counter = Process_Counter(reduce(lambda x, y : x + y, [len(x) for x in _source_datas.values()], 0))

            for key, pairs in _source_datas.items():
                for (main, sub) in pairs:
                    materials = create_materials(main, sub)
                    material_datas[key] += materials
                    counter.add(item="fix", count=len(materials))
                    counter.add()
                counter.print_sequence("[{0}] Create material datas".format(self.get_option()))
            counter.print_result("[{0}] Create material datas".format(self.get_option()))

            return material_datas

        def _except_materials(self, neighbors, materials):
            """
            - 1) 좌/우 혼선 해당 분할정보 제거

            * [merge, branch] 제외
            * Unstable 에서의 제외 : 이웃관계 유지용
            * Unstable 이후의 제외 : centerLine 왜곡 방지용
            - 2) 병합조건 만족 분할정보 제거 (이웃관계 유지)
                - 종료 A1 동일
                - 시작 폭 기준 이하
            
            - 3) 분기조건 만족 분할정보 제거 (이웃관계 유지)
                - 시작 A1 동일
                - 종료 폭 기준 이하
            - 4) 미교차 분할정보 제거
            """

            def except_merge(materials):
                """
                병합 쌍에 해당하는 분할정보 제외
                """

                def check_merge(mat, main, sub):
                    # - ToNode 일치
                    if main.ToNodeID == sub.ToNodeID:
                        # - 시작 좌표간 간격이 최대 도로폭 이하
                        if calc_distance(main.points[0], sub.points[0]) < self.__class__.road_max:
                            return True
                    return False

                remove_list = []

                for mat in materials:
                    main = Shape.get_post("A2", mat["main"])
                    sub = Shape.get_post("A2", mat["sub"])
                    if check_merge(mat, main, sub):
                        remove_list.append(mat)

                materials = [mat for mat in materials if mat not in remove_list]

                return materials

            def except_branch(materials):
                """
                분기 쌍에 해당하는 분할정보 제외
                """

                def check_branch(mat, main, sub):
                    # - FromNode 일치
                    if main.FromNodeID == sub.FromNodeID:
                        # - 종료 좌표간 간격이 최대 도로폭 이하
                        if calc_distance(main.points[-1], sub.points[-1]) < self.__class__.road_max:
                            return True
                    return False

                remove_list = []

                for mat in materials:
                    main = Shape.get_post("A2", mat["main"])
                    sub = Shape.get_post("A2", mat["sub"])
                    if check_branch(mat, main, sub):
                        remove_list.append(mat)

                materials = [mat for mat in materials if mat not in remove_list]

                return materials

            materials = super(self.__class__, self)._except_materials(neighbors, materials)
            materials = self.except_cross_type(neighbors, materials)
            materials = except_merge(materials)
            materials = except_branch(materials)

            return materials

        # --------------------------------------------------

        def _check_unstable(self, main, sub):
            
            # 1) 시작/종료 A1 일치하는 경우 
            # if any([getattr(main, item) == getattr(sub, item) for item in ["FromNodeID", "ToNodeID"]]):
            #     return True

            # 2) 근접/초과 영역 검출 시
            unstable_limit = 1.0
            unstable_length = 0.0

            for main_seg in [main.points[index:index+2] for index in range(len(main.points)-1)]:
                dist = calc_distance_from_line(main_seg, get_closest_point(get_mid(main_seg[0], main_seg[-1]), sub.points))
                if not (self.__class__.road_min <= dist <= self.__class__.road_max):
                    unstable_length += calc_distance(main_seg[0], main_seg[-1])
                    if unstable_length > unstable_limit:
                        return True
            
            return False

        def _set_relation(self, parent_link, new_link, materials, data_list, data_index):
            
            def check_target(parent_link, material):
                
                main_id = material["main"]
                sub_id = material["sub"]

                if parent_link.ID in [main_id, sub_id]:
                    for side_index in [0, 1]:
                        item = ["L_LinkID", "R_LinkID"][side_index]
                        side_id = getattr(parent_link, item)
                        if side_id in [main_id, sub_id]:
                            return True                        

                return False

            # 1. 분할정보 정렬 
            # - 분할정보는 기본 정렬되지 않은 상태
            # - 정렬된 좌표목록(data_list)을 기준으로 정렬
            _materials = []
            for data in data_list:
                material = next(mat for mat in materials if materials.index(mat) == data["index"])
                _materials.append(material)
            materials = _materials

            # 2. 분할정보 추출
            # - 현 A2(parent_link) 에 해당하는 분할정보만 추출
            _materials = []
            for mat in materials:
                if check_target(parent_link, mat):
                    _materials.append(mat)

            # 3. 현 좌표목록(new_link)에 해당하는 분할정보 추출
            if len(_materials) > 0:

                # - upper : 현 좌표목록 순번 이상의 분할정보
                # - lower : 현 좌표목록 순번 미만의 분할정보                
                (upper, lower) = ([], [])
                for mat in _materials:
                    index = materials.index(mat)
                    upper.append(mat) if index >= data_index else lower.append(mat)
                
                # - 1) 현 좌표목록 순번 이상의 분할정보 중에서 가장 빠른 순번 선택
                if len(upper) > 0:
                    material = upper[0]
                    prev_stat = material["prev_stat"]
                # - 2) 현 좌표목록 순번 미만의 분할정보 중에서 가장 느린 순번 선택
                else:
                    material = lower[-1]
                    # - 순번 미만의 경우 Unstable 여부를 반대로 설정한다.
                    prev_stat = not material["prev_stat"]

                # - 현 좌표목록이 Unstable 에 해당 시
                if prev_stat:
                    for item in ["L_LinkID", "R_LinkID"]:
                        # - Main - Sub 관계에 해당하는 이웃관계 해소
                        if getattr(parent_link, item) in [material["main"], material["sub"]]:
                            new_link.replace(**{item : -1})

        # --------------------------------------------------

        def _save_source(self, source_datas):

            map = Convert.create_map()
            
            for pairs in source_datas.values():
                for (main, sub) in pairs:
                    for link in [main, sub]:
                        line3d = Convert.convert_to_lineString3d(link.points)
                        line3d.attributes["ID"] = str(link.ID)
                        map.add(line3d)
                    map.add(Convert.convert_to_lineString3d([
                        get_point_on_points(main.points, None, division=2),
                        get_point_on_points(sub.points, None, division=2)
                    ]))

            Convert.save_map("Shape_Parser", "source.osm", map, sub_dir="{0}".format(self.get_option()))

        def _save_material(self, material_datas):

            map = Convert.create_map()

            for materials in material_datas.values():
                for index, material in enumerate(materials):
                    if material.get("parse_line") != None:
                        line3d = Convert.convert_to_lineString3d(material["parse_line"])
                        line3d.attributes["index"] = str(index)
                        line3d.attributes["main"] = str(material["main"])
                        line3d.attributes["sub"] = str(material["sub"])
                        line3d.attributes["prev"] = str(material["prev_stat"])
                    else:
                        main = Shape.get_post("A2", material["main"])
                        sub = Shape.get_post("A2", material["sub"])
                        line = [
                            get_point_on_points(main.points, division=2),
                            get_point_on_points(sub.points, division=2),
                        ]
                        line3d = Convert.convert_to_lineString3d(line)
                        line3d.attributes["ID"] = "{0} : {1}".format(main.ID, sub.ID)
                    map.add(line3d)

            Convert.save_map("Shape_Parser", "material.osm", map, sub_dir="{0}".format(self.get_option()))

    class Parse_Length(Parse_Process):

        max_length = 100.0

        def extract_source_datas(self, neighbor_datas):

            source_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(neighbor_datas))

            for key, neighbors in neighbor_datas.items():
                min_link = A2.select_min(neighbors)
                min_length = min_link.Length
                if self._check_length(min_length):
                    source_datas[key] = min_link 
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[{0}] Extract source datas".format(self.get_option()))
            counter.print_result("[{0}] Extract source datas".format(self.get_option()))

            return source_datas

        def create_material_datas(self, source_datas):

            def create_materials(min_link):

                materials = []

                neighbors = A2.get_neighbors(min_link.ID)
                standard = A2.get_main(min_link.ID)

                parse_length = self.__class__.max_length / 2.0
                min_length = min_link.Length
                count = int(min_length / parse_length)

                for index in range(count - 1):

                    length = (index + 1) * (min_length / float(count))
                    parse_point = get_point_on_points(min_link.points, length=length)
                    seg = get_closest_segment(parse_point, standard.points)
                    parse_line = get_ortho_line(parse_point, start=seg[0], end=seg[-1])
                    
                    material = {
                        "parse_line" : parse_line,
                    }
                    materials.append(material)

                return materials

            material_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(source_datas))

            for key, min_link in source_datas.items():
                materials = create_materials(min_link)
                material_datas[key] += materials
                counter.add(item="fix", count=len(materials))
                counter.add()
                counter.print_sequence("[{0}] Create material datas".format(self.get_option()))
            counter.print_result("[{0}] Create material datas".format(self.get_option()))

            return material_datas

        def _except_materials(self, neighbors, materials):
            """
            - 1) 병합 + 수렴영역 해당 시 제외
            - 2) 분기 + 발산영역 해당 시 제외
            """

            materials = super(self.__class__, self)._except_materials(neighbors, materials)
            materials = self.except_merge(neighbors, materials)
            materials = self.except_branch(neighbors, materials)

            return materials

        # --------------------------------------------------

        def _check_length(self, min_length):
            return True if min_length > self.__class__.max_length else False

        def _set_relation(self, parent_link, new_link, materials, data_list, data_index):
            pass

        # --------------------------------------------------

        def _save_source(self, source_datas):

            map = Convert.create_map()
            
            for min_link in source_datas.values():
                map.add(Convert.convert_to_lineString3d(min_link.points))

            Convert.save_map("Shape_Parser", "source.osm", map, sub_dir="{0}".format(self.get_option()))

        def _save_material(self, material_datas):

            map = Convert.create_map()

            for materials in material_datas.values():
                for material in materials:
                    map.add(Convert.convert_to_lineString3d(material["parse_line"]))
            
            Convert.save_map("Shape_Parser", "material.osm", map, sub_dir="{0}".format(self.get_option()))

    class Parse_Lane(Parse_Process):

        def extract_source_datas(self, neighbor_datas):
            
            def check_lane(link):

                default_type = 999

                (left_lanes, right_lanes) = A2.get_lanes(link.ID)
                lane_types = [lane.Type for lane in left_lanes + right_lanes]

                if len(left_lanes + right_lanes) < 1:
                    return False
                elif len(lane_types) < 2:
                    if lane_types[0] == default_type:
                        return False

                return True

            source_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(neighbor_datas))

            for key, neighbors in neighbor_datas.items():
                for link in neighbors:
                    if check_lane(link):
                        source_datas[key].append(link)
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[{0}] Extract source datas".format(self.get_option()))
            counter.print_result("[{0}] Extract source datas".format(self.get_option()))

            return source_datas

        def create_material_datas(self, source_datas):

            def create_materials(links):

                def create(link, standard):
                    
                    _materials = []

                    (left_lanes, right_lanes) = A2.get_lanes(link.ID)

                    for lane in (left_lanes + right_lanes):
                        for index in [0, -1]:
                            # 1) B2 종단점 분할선 추출
                            lane_p = lane.points[index]
                            seg = get_closest_segment(lane_p, standard.points)
                            parse_line = get_ortho_line(lane_p, start=seg[0], end=seg[-1])

                            # 2) 분할선과 기준 A2 간 교차정보 추출
                            intersect_p = get_intersection_on_points(parse_line, standard.points)
                            # - 교차 시
                            if intersect_p != None:
                                # - 분할선 평행이동
                                mid = get_mid(parse_line[0], parse_line[-1])
                                parse_line = move_line(parse_line, start=mid, end=intersect_p)
                                _material = {
                                    "parse_line" : parse_line,
                                    "lane" : lane.ID,
                                    "link" : link.ID,
                                    "index" : index,
                                }
                                _materials.append(_material)

                    return _materials 

                def sort(_materials, standard):

                    __materials = []

                    for seg in [standard.points[index:index+2] for index in range(len(standard.points)-1)]:
        
                        _list = []
                        for mat in _materials:
                            if mat not in __materials:
                                if check_point_on_line(seg, mat["parse_line"][1]):
                                    _list.append(mat)

                        _list = sorted(_list, key=lambda x : calc_distance(x["parse_line"][1], seg[0]))
                        __materials += _list

                    if len(__materials) < len(_materials):
                        warning_print("Sorting Exception")

                    return __materials

                def deduplicate(_materials, standard):
                    
                    __materials = []

                    stack = []

                    for _material in _materials:
                    
                        # - stack 이 빈 경우 (중복 상태 X)
                        if len(stack) < 1:
                            # - 추가
                            __materials.append(_material)
                        else:
                            # - 최근접 검사
                            seg = get_closest_segment(_material["parse_line"][1], standard.points)

                            for mat in stack:
                                prev_dist = calc_distance_from_line(seg, mat["parse_line"][1])
                                curr_dist = calc_distance_from_line(seg, _material["parse_line"][1])
                                if prev_dist < curr_dist:
                                    break
                            # - stack 전체에서 최근접 판정 시
                            else:
                                # - 추가
                                __materials.append(_material)

                        # - 시작 좌표에 해당
                        if _material["index"] == 0:
                            stack.append(_material)
                        # - 종료 좌표에 해당
                        else:
                            # - 동일한 B2 기반 분할정보를 stack 에서 제거
                            mat = next((x for x in stack if x["lane"] == _material["lane"]), None)
                            if mat != None:
                                index = stack.index(mat)
                                stack.pop(index)

                    return __materials

                materials = []

                standard = A2.get_main(links[0].ID)

                for link in links:
                    _materials = create(link, standard)
                    _materials = sort(_materials, standard)
                    _materials = deduplicate(_materials, standard)
                    materials += _materials

                materials = sort(materials, standard)

                return materials

            material_datas = defaultdict(lambda : [])

            counter = Process_Counter(len(source_datas))

            for key, links in source_datas.items():
                materials = create_materials(links)
                material_datas[key] += materials
                counter.add(item="fix", count=len(materials))
                counter.add()
                counter.print_sequence("[{0}] Create material datas".format(self.get_option()))
            counter.print_result("[{0}] Create material datas".format(self.get_option()))

            return material_datas

        def _except_materials(self, neighbors, materials):
            """
            - 1) 좌/우 혼선 해당 분할정보 제거
            - 2) 병합 + 수렴영역 해당 시 제외
            - 3) 분기 + 발산영역 해당 시 제외
            """

            materials = super(self.__class__, self)._except_materials(neighbors, materials)
            materials = self.except_cross_type(neighbors, materials)
            materials = self.except_merge(neighbors, materials)
            materials = self.except_branch(neighbors, materials)

            return materials

        # --------------------------------------------------

        def _check_length(self, min_length):
            return True if min_length > self.__class__.max_length else False

        def _set_relation(self, parent_link, new_link, materials, data_list, data_index):
            pass

        # --------------------------------------------------

        def _save_source(self, source_datas):

            map = Convert.create_map()
            
            for links in source_datas.values():
                for link in links:
                    map.add(Convert.convert_to_lineString3d(link.points))

            Convert.save_map("Shape_Parser", "source.osm", map, sub_dir="{0}".format(self.get_option()))

        def _save_material(self, material_datas):

            map = Convert.create_map()

            for materials in material_datas.values():
                for index, material in enumerate(materials):
                    line3d = Convert.convert_to_lineString3d(material["parse_line"], link_id=index+1)
                    line3d.attributes["Link"] = str(material["link"])
                    map.add(line3d)
            
            Convert.save_map("Shape_Parser", "material.osm", map, sub_dir="{0}".format(self.get_option()))

        # --------------------------------------------------

        def _extract_lane_types(self):

            link_datas = Shape.get_post_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link_id, link in link_datas.items():

                lane_type = [999, 999]

                mid = get_point_on_points(link.points, division=2)
                seg = get_inner_segment(mid, link.points)
                ortho_line = get_ortho_line(mid, start=seg[0], end=seg[-1])

                (left_lanes, right_lanes) = A2.get_lanes(link_id)
                for index, side_lanes in enumerate([left_lanes, right_lanes]):

                    min_lane = None
                    min_dist = float("inf")
                    for lane in side_lanes:
                        intersect_p = get_intersection_on_points(ortho_line, lane.points)
                        if intersect_p != None:
                            dist = calc_distance_from_line(seg, intersect_p)
                            if dist < min_dist:
                                min_lane = lane
                                min_dist = dist

                    if min_lane != None:
                        lane_type[index] = min_lane.Type
                        counter.add(item="fix")

                for side_index in [0, 1]:
                    A2.set_lane_type(link_id, lane_type[side_index], side_index)

                counter.add()
                counter.print_sequence("[{0}] Extract lane types".format(self.get_option()))
            counter.print_result("[{0}] Extract lane types".format(self.get_option()))

        def _revise_lane_types(self):
            
            def revise_diff():

                def get_color_order(type_code):
                    """
                    황색 / 청색 / 백색 우선순위 
                    """

                    color_code = type_code / 100

                    order = {
                        1 : 1, # 황색
                        2 : 3, # 청색
                        3 : 2, # 백색
                        9 : 4, # 가상
                    }.get(color_code)

                    return order

                def get_class_order(type_code):
                    """
                    겹선 / 단선 우선순위
                    """

                    class_code = (type_code % 100) / 10

                    order = {
                        1 : 2, # 겹선
                        2 : 1, # 단선
                        9 : 3, # 가상
                    }.get(class_code)

                    return order

                def get_line_order(type_code):
                    """
                    실선 / 혼선 / 점선 우선순위
                    """

                    line_code = (type_code % 10)

                    order = {
                        1 : 1, # 실선
                        2 : 3, # 점선
                        3 : 2, # 좌점혼선
                        4 : 2, # 우점혼선
                        9 : 4, # 가상
                    }.get(line_code)

                    return order

                def select_type(curr_type, side_type):
                    """
                    color, class, line 은 값이 낮을수록 우선순위 높음
                    """

                    curr_color = get_color_order(curr_type)
                    side_color = get_color_order(side_type)

                    if curr_color != side_color:
                        selected = curr_type if curr_color < side_color else side_type
                    else:
                        curr_class = get_class_order(curr_type)
                        side_class = get_class_order(side_type)
                        
                        if curr_class != side_class:
                            selected = curr_type if curr_class < side_class else side_type
                        else:
                            curr_line = get_line_order(curr_type)
                            side_line = get_line_order(side_type)

                            if curr_line != side_line:
                                selected = curr_type if curr_line < side_line else side_type
                            else:
                                selected = 999
                    
                    return selected

                def save_error(error):
                    
                    if len(error) < 1:
                        return 

                    map = Convert.create_map()

                    for link_id, side_ids in error.items():
                        
                        link = Shape.get_post("A2", link_id)
                        if link != None:
                            line3d = Convert.convert_to_lineString3d(link.points)
                            map.add(line3d)

                            for side_id in side_ids:
                                side_link = Shape.get_post("A2", side_id)
                                if side_link != None:
                                    line3d = Convert.convert_to_lineString3d(side_link.points)
                                    map.add(line3d)

                                    line = [
                                        get_point_on_points(link.points, division=2),
                                        get_point_on_points(side_link.points, division=2)
                                    ]
                                    line3d = Convert.convert_to_lineString3d(line)
                                    map.add(line3d)

                    Convert.save_map("Shape_Parser", "revise_diff(error).osm", map, sub_dir="lane")

                link_datas = Shape.get_post_datas("A2")
                counter = Process_Counter(len(link_datas))

                record = defaultdict(lambda : False)
                error = defaultdict(lambda : [])

                for link_id, link in link_datas.items():
                    if not record[link_id]:
                        (left_link, right_link) = A2.get_side(link)
                        for index, side_link in enumerate([left_link, right_link]):
                            if side_link != None:
                                curr_type = A2.get_lane_type(link_id)[index]
                                side_type = A2.get_lane_type(side_link.ID)[1-index]

                                # - 이웃 간 B2 Type 이 다르게 추출된 경우
                                if curr_type != side_type:
                                    # - 우선순위에 따라 통합
                                    selected = select_type(curr_type, side_type)
                                    if selected == side_type:
                                        A2.set_lane_type(link_id, side_type, index)
                                    else:
                                        A2.set_lane_type(side_link.ID, curr_type, 1-index)    

                                    # - 가상 차선이 포함되지 않은 경우 = 서로다른 차선 추출 (예외)
                                    if 999 not in [curr_type, side_type]:
                                        warning_print("({0}) {1} : {2}".format(index, link.ID, curr_type))
                                        warning_print("({0}) {1} : {2}".format(index, side_link.ID, side_type))
                                        error[link_id].append(side_link.ID)
                                        counter.add(item="warn")

                                    counter.add(item="fix")
                                record[side_link.ID] = True
                        record[link_id] = True
                    counter.add()
                    counter.print_sequence("[{0}] Revise lane diff".format(self.get_option()))
                counter.print_result("[{0}] Revise lane diff".format(self.get_option()))

                save_error(error)

            def revise_neighbor():
                """
                좌/우 이웃한 A2 사이 차선이 가상 차선인 경우 차선변경을 위해 점선으로 설정
                - 1) 이웃한 경우
                - 2) 가상 차선인 경우
                """

                link_datas = Shape.get_post_datas("A2")
                counter = Process_Counter(len(link_datas))

                # - 수정 후 차선타입
                target_type = 312

                for link_id, link in link_datas.items():
                    lane_type = A2.get_lane_type(link_id)
                    for side_index, side_link in enumerate(A2.get_side(link)):
                        # 1) 이웃한 경우
                        if side_link != None:
                            side_type = lane_type[side_index]
                            # 2) 가상 차선인 경우
                            if side_type == 999:
                                A2.set_lane_type(link_id, target_type, side_index)
                                A2.set_lane_type(side_link.ID, target_type, 1-side_index)                    
                                counter.add(item="fix") 
                    counter.add()
                    counter.print_sequence("[{0}] Revise lane neighbor".format(self.get_option()))
                counter.print_result("[{0}] Revise lane neighbor".format(self.get_option()))

            def revise_intersection():
                """
                이웃한 A2 사이 차선 수정
                - 1) 교차로인 경우
                - 2) 이웃한 경우
                - 3) 사이 차선이 점선인 경우
                -> 교차로 진입 시 차선변경을 금지하기 위해 차선 수정
                """

                link_datas = Shape.get_post_datas("A2")
                counter = Process_Counter(len(link_datas))

                # - 수정 후 차선타입
                target_type = 999

                for link_id, link in link_datas.items():
                    # 1) 교차로인 경우
                    if A2.check_intersection(link_id):
                        lane_type = A2.get_lane_type(link_id)
                        for side_index, side_link in enumerate(A2.get_side(link)):
                            # 2) 이웃한 경우
                            if side_link != None:
                                side_type = lane_type[side_index]
                                # 3) 사이 차선이 점선인 경우
                                if side_type % 10 in [2, 3, 4]:
                                    A2.set_lane_type(link_id, target_type, side_index)
                                    A2.set_lane_type(side_link.ID, target_type, 1-side_index)                    
                                    counter.add(item="fix") 
                    counter.add()
                    counter.print_sequence("[{0}] Revise lane intersection".format(self.get_option()))
                counter.print_result("[{0}] Revise lane intersection".format(self.get_option()))

            def validate_diff():

                line3ds = []

                link_datas = Shape.get_post_datas("A2")
                for link in link_datas.values():
                    lane_type = A2.get_lane_type(link.ID)
                    for side_index, side_link in enumerate(A2.get_side(link)):
                        if side_link != None:
                            side_lane_type = A2.get_lane_type(side_link.ID)
                            if lane_type[side_index] != side_lane_type[1-side_index]:
                                line3d = Convert.convert_to_lineString3d(link.points)
                                line3d.attributes["subtype"] = str(lane_type[side_index])
                                line3ds.append(line3d)
                                line3d = Convert.convert_to_lineString3d(side_link.points)
                                line3d.attributes["subtype"] = str(side_lane_type[1-side_index])
                                line3ds.append(line3d)
                                line3d = Convert.convert_to_lineString3d([
                                    get_point_on_points(link.points, division=2),
                                    get_point_on_points(side_link.points, division=2),
                                ])
                                line3ds.append(line3d)
                
                if len(line3ds) > 0:
                    warning_print("Revise diff : Failed ({0})".format(len(line3ds)))
                    map = Convert.create_map()
                    for line3d in line3ds:
                        map.add(line3d)
                    Convert.save_map("Shape_Parser", "revise_diff.osm", map, sub_dir="lane")
        
            def validate_neighbor():

                line3ds = []

                link_datas = Shape.get_post_datas("A2")
                for link in link_datas.values():
                    lane_type = A2.get_lane_type(link.ID)
                    for side_index, side_link in enumerate(A2.get_side(link)):
                        if side_link != None:
                            side_lane_type = A2.get_lane_type(side_link.ID)
                            if lane_type[side_index] == side_lane_type[1-side_index]:
                                if lane_type[side_index] == 999:
                                    line3d = Convert.convert_to_lineString3d(link.points)
                                    line3d.attributes["subtype"] = str(lane_type[side_index])
                                    line3ds.append(line3d)
                                    line3d = Convert.convert_to_lineString3d(side_link.points)
                                    line3d.attributes["subtype"] = str(side_lane_type[1-side_index])
                                    line3ds.append(line3d)
                                    line3d = Convert.convert_to_lineString3d([
                                        get_point_on_points(link.points, division=2),
                                        get_point_on_points(side_link.points, division=2),
                                    ])
                                    line3ds.append(line3d)
                
                if len(line3ds) > 0:
                    warning_print("Revise neighbor : Failed ({0})".format(len(line3ds)))
                    map = Convert.create_map()
                    for line3d in line3ds:
                        map.add(line3d)
                    Convert.save_map("Shape_Parser", "revise_neighbor.osm", map, sub_dir="lane")
        
            def validate_intersection():

                line3ds = []

                link_datas = Shape.get_post_datas("A2")
                for link in link_datas.values():
                    if A2.check_intersection(link.ID):
                        lane_type = A2.get_lane_type(link.ID)
                        for side_index, side_link in enumerate(A2.get_side(link)):
                            if side_link != None:
                                side_lane_type = A2.get_lane_type(side_link.ID)
                                if lane_type[side_index] == side_lane_type[1-side_index]:
                                    if lane_type[side_index] % 10 in [2, 3, 4]:
                                        line3d = Convert.convert_to_lineString3d(link.points)
                                        line3d.attributes["subtype"] = str(lane_type[side_index])
                                        line3ds.append(line3d)
                                        line3d = Convert.convert_to_lineString3d(side_link.points)
                                        line3d.attributes["subtype"] = str(side_lane_type[1-side_index])
                                        line3ds.append(line3d)
                                        line3d = Convert.convert_to_lineString3d([
                                            get_point_on_points(link.points, division=2),
                                            get_point_on_points(side_link.points, division=2),
                                        ])
                                        line3ds.append(line3d)
                
                if len(line3ds) > 0:
                    warning_print("Revise intersection : Failed ({0})".format(len(line3ds)))
                    map = Convert.create_map()
                    for line3d in line3ds:
                        map.add(line3d)
                    Convert.save_map("Shape_Parser", "revise_intersection.osm", map, sub_dir="lane")
        
            revise_diff()
            validate_diff()

            revise_neighbor()
            validate_neighbor()

            revise_intersection()
            validate_intersection()

        # --------------------------------------------------

        def execute(self):
            super(self.__class__, self).execute()
            self._extract_lane_types()
            self._revise_lane_types()

    @classmethod
    def execute(cls):
        cls.Check_Multiple().execute()
        cls.Parse_Unbalance().execute()
        cls.Parse_Unstable().execute()
        cls.Parse_Length().execute()
        cls.Parse_Lane().execute()


class Shape_Parser(Module):
    
    def do_process(cls, *args, **kwargs):
        
        Shape.create_post()
        A2_Parser.execute()

        return True
