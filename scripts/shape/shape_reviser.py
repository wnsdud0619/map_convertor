#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import reduce

from scripts.shape.shape_data import (
    Shape,
    A2,
    A3,
    B2,
    Quad,
    Convert,
)
from scripts.shape.shape_module import Module
from scripts.functions.coordinate_functions import (
    get_center,
    check_same,
    calc_length,
    calc_distance,
    get_closest_point,
    deduplicate_points,
    calc_parallel,
    get_closest_quad_point,
    get_point_on_points,
    select_straight, 
    correct_misdirected,
    check_curve_intersect,
    check_is_left,
    get_mid,
    check_inside,
    simplify_polygon,
)
from scripts.functions.print_functions import (
    Process_Counter, 
    log_print,
    warning_print,
)


class A1_Reviser():
    """
    A1 관련 정보 보완
    """

    class Remove_Isolated():
        """
        타 shape data(A2_LINK) 와 어떤 관계도 존재하지 않는 A1_NODE 제거 
        """

        def execute(self):            
            before = len(Shape.get_shape_datas("A1"))
            counter = Process_Counter(before)

            # 1. A2 에 등록된 A1 기록
            record = dict()
            for link in Shape.get_shape_datas("A2").values():
                record[link.FromNodeID] = True
                record[link.ToNodeID] = True

            # 2. 기록되지 않은 A1 을 제외한 A1 목록 갱신
            # - 제외된 목록 생성
            node_datas = {node.ID : node for node in Shape.get_shape_datas("A1").values() if record.get(node.ID) != None}
            # - 목록 갱신
            Shape.set_shape_datas("A1", node_datas)

            after = len(Shape.get_shape_datas("A1"))
            counter.add(item="fix", count=before - after)
            counter.add(count=before)
            counter.print_result("[A1] Remove isolated")

    class Add_Missed():
        """
        A2_LINK 에 등록되었으나 A1_NODE 목록에 존재하지 않는 A1_NODE 신규 생성
        """

        def execute(self):
            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))
            
            node_datas = Shape.get_shape_datas("A1")
            _AdminCode = node_datas.values()[0].AdminCode if len(node_datas) > 0 else 0
            _NodeType = 99

            for link in link_datas.values():
            
                if Shape.get_shape("A1", link.FromNodeID) == None:
                    instance = (Shape.Instance()
                        .replace(ID=link.FromNodeID)
                        .replace(AdminCode=_AdminCode)
                        .replace(NodeType=_NodeType)
                        .replace(points=[link.points[0]])
                    )
                    Shape.set_shape("A1", instance)
                    counter.add(item="fix")
            
                if Shape.get_shape("A1", link.ToNodeID) == None:
                    instance = (Shape.Instance()
                        .replace(ID=link.ToNodeID)
                        .replace(AdminCode=_AdminCode)
                        .replace(NodeType=_NodeType)
                        .replace(points=[link.points[-1]])
                    )
                    Shape.set_shape("A1", instance)
                    counter.add(item="fix")
            
                counter.add()
                counter.print_sequence("[A1] Add missed")

            counter.print_result("[A1] Add missed")

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.Remove_Isolated().execute()
        cls.Add_Missed().execute()


class A2_Reviser():
    """
    A2 관련 정보 보완
    """

    class Correct_Oneway_Relation():
        """
        A2 단방향 관계 제거
        """

        def execute(self):

            record = dict()

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                
                left_link = Shape.get_shape("A2", link.L_LinkID)
                if left_link != None:
                    if left_link.R_LinkID != link.ID:
                        link.replace(L_LinkID=-1)
                        left_link.replace(R_LinkID=-1)
                        counter.add(item="fix")
                        record[len(record)] = (link, left_link)

                right_link = Shape.get_shape("A2", link.R_LinkID)
                if right_link != None:
                    if right_link.L_LinkID != link.ID:
                        link.replace(R_LinkID=-1)
                        right_link.replace(L_LinkID=-1)
                        counter.add(item="fix")
                        record[len(record)] = (link, right_link)
                
                counter.add()
                counter.print_sequence("[A2] Correct one-way relation")
            counter.print_result("[A2] Correct one-way relation")

            map = Convert.create_map()
            for (link, side_link) in record.values():
                map.add(Convert.convert_to_lineString3d(link.points))
                map.add(Convert.convert_to_lineString3d(side_link.points))
            Convert.save_map("Shape_Reviser", "oneway_relation.osm", map, sub_dir="A2")

    class Break_Intersect_Relation():
        """
        교차 이웃 A2 관계 해소
        """

        def execute(self):

            def check_branch(link, side_link):
                """
                시작/종료 A1 의 동일 여부 반환
                """

                for item in ["FromNodeID", "ToNodeID"]:
                    if getattr(link, item) == getattr(side_link, item):
                        return True
                return False

            def save(record):
            
                map = Convert.create_map()

                for (main, sub) in record:

                    line3d = Convert.convert_to_lineString3d(main.points)
                    line3d.attributes["ID"] = str(main.ID)
                    line3d.attributes["Role"] = "main"
                    map.add(line3d)

                    line3d = Convert.convert_to_lineString3d(sub.points)
                    line3d.attributes["ID"] = str(sub.ID)
                    line3d.attributes["Role"] = "sub"
                    map.add(line3d)

                    line = [
                        get_point_on_points(main.points, division=2),
                        get_point_on_points(sub.points, division=2)
                    ]
                    line3d = Convert.convert_to_lineString3d(line)
                    map.add(line3d)

                Convert.save_map("Shape_Reviser", "[A2] Break Intersect.osm", map, sub_dir="A2")

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            record = []

            for link in link_datas.values():
                for side_index in [0, 1]:
                    side_id = getattr(link, ["L_LinkID", "R_LinkID"][side_index])
                    side_link = Shape.get_shape("A2", side_id)
                    if side_link != None:
                        if not check_branch(link, side_link):
                            if check_curve_intersect(link.points, side_link.points):
                                link.replace(**{["L_LinkID", "R_LinkID"][side_index] : -1})
                                side_link.replace(**{["L_LinkID", "R_LinkID"][1-side_index] : -1})
                                counter.add(item="fix")
                                record.append((link, side_link))
                counter.add()
                counter.print_sequence("[A2] Break intersect relation")
            counter.print_result("[A2] Break intersect relation")

            save(record)

    class Break_Close_Relation():
        """
        근접 조건을 만족하는 A2 이웃관계 제거
        - 1) 평행영역 기준 이하
        """

        def execute(self):

            def check_close(link, side_link, check_length=4.0):
                
                # 1. From / To 동일 검사
                # - From / To 중 일치 개수가 1개 이상인 경우 False 반환
                if any([getattr(link, item) == getattr(side_link, item) for item in ["FromNodeID", "ToNodeID"]]):
                    return False

                # 2. main - sub 구분 (보다 직선에 가까운)
                (main, sub) = A2.classify_pair(link, side_link)

                # 3. 평행 거리 검사
                # - main - sub 간 평행영역의 총 길이가 기준거리(check_length) 미만인 경우 근접 판정
                parallel_length = calc_parallel(main.points, sub.points) 
                if parallel_length < check_length:
                    result = True
                else:
                    result = False


                return result

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                left_link = Shape.get_shape("A2", link.L_LinkID)
                if left_link != None:
                    if check_close(link, left_link):
                        link.replace(L_LinkID=-1)
                        left_link.replace(R_LinkID=-1)
                        counter.add(item="fix")

                right_link = Shape.get_shape("A2", link.R_LinkID)
                if right_link != None:
                    if check_close(link, right_link):
                        link.replace(R_LinkID=-1)
                        right_link.replace(L_LinkID=-1)
                        counter.add(item="fix")

                counter.add()
                counter.print_sequence("[A2] Break close relation")
            counter.print_result("[A2] Break close relation")

    class Add_Missed_Node():
        """
        A2 에 등록된 A1 이 없는 경우 신규 등록
        """

        def execute(self):

            def get_AdminCode():
                """
                권역코드(AdminCode) 검색
                """
                node_datas = Shape.get_shape_datas("A1")
                AdminCode = next((x.AdminCode for x in node_datas.values() if x.AdminCode != None), None)
                return AdminCode

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            AdminCode = get_AdminCode()
            NodeType = 99

            for link in link_datas.values():
                # - From A1 이 없는 경우
                if Shape.get_shape("A1", link.FromNodeID) == None:
                    # 1) 신규 A1 생성
                    instance = (Shape.Instance()
                        .replace(ID=link.FromNodeID)
                        .replace(AdminCode=AdminCode)
                        .replace(NodeType=NodeType)
                        .replace(points=[link.points[0]])
                    )
                    Shape.set_shape("A1", instance)
                    counter.add(item="fix")

                # - To A1 이 없는 경우
                if Shape.get_shape("A1", link.ToNodeID) == None:
                    # 1) 신규 A1 생성
                    instance = (Shape.Instance()
                        .replace(ID=link.ToNodeID)
                        .replace(AdminCode=AdminCode)
                        .replace(NodeType=NodeType)
                        .replace(points=[link.points[-1]])
                    )
                    Shape.set_shape("A1", instance)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Add missed node")
            counter.print_result("[A2] Add missed node")

    class Correct_Link_Pointer():
        """
        존재하지 않는 A2 참조 Shape 정보 수정
        """

        def execute(self):

            domain = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C3", "C4", "C5", "C6"]
            counter = Process_Counter(reduce(lambda x, y : x + y, [len(Shape.get_shape_datas(shape_type)) for shape_type in domain], 0))

            # 1. A2 필드목록 지정
            columns = ["LinkID", "L_LinkID", "R_LinkID"]                

            for shape_type in domain:
                pointers = [column for column in columns if column in Shape.get_columns(shape_type)]
                for shape in Shape.get_shape_datas(shape_type).values():
                    for pointer in pointers:
                        if getattr(shape, pointer) != -1 and Shape.get_shape("A2", getattr(shape, pointer)) == None:
                            shape.replace(**{pointer : -1})
                            counter.add(item="fix")
                counter.add(count=len(Shape.get_shape_datas(shape_type)))
                counter.print_sequence("[A2] Correct link pointer")
            counter.print_result("[A2] Correct link pointer")

    class Correct_Length():
        """
        A2.Length 보완
        """

        def execute(self):

            def check_length(link):
                try:
                    1 / float(link.Length)
                # - None or 0 값인 경우 False
                except (ValueError, ZeroDivisionError):
                    return False
                else:
                    return True

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                # 1) Length 보완필요여부 검사
                if not check_length(link):
                    # 2) Length 보완
                    link.replace(Length=calc_length(link.points))
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Correct length")
            counter.print_result("[A2] Correct length")

    class Deduplicate_points():
        """
        A2 중복좌표 제거
        """

        def execute(self):

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                # 1) 중복좌표제거 수행
                points = deduplicate_points(link.points)
                # - 중복여부 검사
                if len(points) != len(link.points):
                    # 2) 좌표목록 갱신
                    link.replace(points=points)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Deduplicate points")
            counter.print_result("[A2] Deduplicate points")

    class Loosen_Dense_Points():
        """
        A2 근접좌표 제거
        """            

        def execute(self):

            def get_arranged(points, close_dist=.1):
                new_points = points[:1]
                for index in range(1, len(points) - 1):
                    prev = new_points[-1]
                    curr = points[index]
                    if calc_distance(prev, curr) > close_dist:
                        new_points.append(curr)
                new_points.append(points[-1])
                return new_points

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                # - 좌표 2개 초과인 A2 대상
                if len(link.points) > 2:
                    # 1) 근접좌표 제거
                    new_points = get_arranged(link.points)
                    # - 제거여부 검사
                    if len(new_points) < len(link.points):
                        # 2) 갱신
                        link.replace(points=new_points)
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Loosen dense points")
            counter.print_result("[A2] Loosen dense points")

    class Correct_Misdirected():
        """
        A2 방향위반 좌표 제거
        """

        def execute(self):

            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link in link_datas.values():
                # 1) 방향위반 좌표 제거 수행
                new_points = correct_misdirected(link.points)
                # - 좌표 변경점 존재 시
                if len(new_points) < len(link.points):
                    # 2) A2 갱신
                    link.replace(points=new_points)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Correct misdirected")
            counter.print_result("[A2] Correct misdirected")

    class Adjust_Gapped():
        """
        A2_LINK 종단 좌표값과 A1_NODE 좌표값이 일치하지 않는 경우 수정
        """

        def execute(self):
            
            def create_relation():
                """
                A1 : ([From A2 list], [To A2 list]) 관계 생성
                """

                relation = defaultdict(lambda : ([], []))

                # - Node ID : ([From Link ID list], [To Link ID list])
                for link in Shape.get_shape_datas("A2").values():
                    relation[link.FromNodeID][1].append(link.ID)
                    relation[link.ToNodeID][0].append(link.ID)

                return relation

            def check_gapped(relation, node_id):
                
                node = Shape.get_shape("A1", node_id)
                (from_link_ids, to_link_ids) = relation.get(node_id)

                for link_id in from_link_ids:
                    from_link = Shape.get_shape("A2", link_id)
                    if not check_same(from_link.points[-1], node.points[0]):
                        return False

                for link_id in to_link_ids:
                    to_link = Shape.get_shape("A2", link_id)
                    if not check_same(to_link.points[0], node.points[0]):
                        return False

                return True

            def get_points(relation, node_id):
                link_points = []

                (from_link_ids, to_link_ids) = relation.get(node_id)
            
                for link_id in from_link_ids:
                    from_link = Shape.get_shape("A2", link_id)
                    link_points.append(from_link.points[-1])
            
                for link_id in to_link_ids:
                    to_link = Shape.get_shape("A2", link_id)
                    link_points.append(to_link.points[0])

                return link_points

            def set_point(point, relation, node_id):
                
                # 1. A1 좌표값 수정
                Shape.get_shape("A1", node_id).replace(points=[point])

                # 2. A2 좌표값 수정
                (from_link_ids, to_link_ids) = relation.get(node_id)
                
                for link_id in from_link_ids:
                    from_link = Shape.get_shape("A2", link_id)
                    from_link.replace(points=from_link.points[:-1] + [point])
                
                for link_id in to_link_ids:
                    to_link = Shape.get_shape("A2", link_id)
                    to_link.replace(points=[point] + to_link.points[1:])

            node_datas = Shape.get_shape_datas("A1")
            counter = Process_Counter(len(node_datas))

            # 1. 관계 테이블 생성
            relation = create_relation()

            # 2. A1 기준 좌표값 조정  
            for node_id in node_datas.keys():
                # 1) 이격 검사
                if not check_gapped(relation, node_id):
                    # 2) From / To Link 종단좌표목록 추출
                    link_points = get_points(relation, node_id)
                    # 3) 평균 좌표 추출
                    average_point = get_center(link_points)
                    # 4) 좌표값 갱신
                    set_point(average_point, relation, node_id)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Adjust gapped")
            counter.print_result("[A2] Adjust gapped")

    class Apply_Schoolzone():
        """
        스쿨존 영역(A3)과 겹치는 주행경로(A2) 최대속도 제한
        """

        def execute(self):
            """
            * A3 영역의 중심을 기준으로 검색
            * A3 영역이 클수록 검색속도 저하
            * A3 영역이 오목한 다각형일수록 검색효율 저하
            """

            # 1. 스쿨존 추출
            section_datas = Shape.get_shape_datas("A3")
            schoolzone_datas = dict()
            for section_id, section in section_datas.items():
                if A3.check_schoolzone(section_id):
                    points = simplify_polygon(section.points, elapse_dist=0.5)
                    schoolzone_datas[section_id] = section

            counter = Process_Counter(len(schoolzone_datas))

            # 2. 스쿨존 내 주행경로 추출
            target_datas = dict()
            total_points = {}
            (quad_tree, quad_table) = Quad.get_quad("A2_edge")                    
            for section_id, section in schoolzone_datas.items():
                section_points = {x[0] : x[1] for x in total_points.items()}
                points = section.points
                center = get_center(points)
                max_dist = max([calc_distance(center, x) for x in points])
                while True:
                    # - 근접 좌표 추출 (중복 제외)
                    closest_p = get_closest_quad_point(center, quad_tree, except_points=section_points)
                    # - 근접 좌표가 존재하는 경우
                    if closest_p != None:
                        # - 근접 범위 내 
                        distance = calc_distance(closest_p, center)
                        if distance <= max_dist:
                            section_points.update({closest_p : True})
                            point2d = (closest_p[0], closest_p[1])
                            link_ids = quad_table[point2d]
                            # - 근접 좌표가 스쿨존 내부에 포함되는 경우
                            if check_inside(closest_p, points):
                                for link_id in link_ids:
                                    link = Shape.get_shape("A2", link_id)
                                    if target_datas.get(link_id) == None:
                                        if link.MaxSpeed > A3.get_speed("school"):
                                            target_datas[link_id] = link
                                            total_points.update({closest_p : True})
                                            counter.add(item="warn")
                                            counter.print_sequence("[A2] Apply schoolzone")
                            continue
                    break
                counter.add()
                counter.print_sequence("[A2] Apply schoolzone")

            # 3. 스쿨존 내 주행경로 수정
            link_datas = Shape.get_shape_datas("A2")
            for link_id, link in link_datas.items():
                if target_datas.get(link_id) != None:
                    link.replace(MaxSpeed=A3.get_speed("school"))
                    counter.add(item="fix")
                    counter.print_sequence("[A2] Apply schoolzone")
            counter.print_result("[A2] Apply schoolzone")
            
            map = Convert.create_map()
            for section_id, section in schoolzone_datas.items():
                points = section.points
                polygon = Convert.convert_to_lineString3d(points)
                map.add(polygon)
            for link_id, link in target_datas.items():
                line3d = Convert.convert_to_lineString3d(link.points)
                line3d.attributes["A2"] = link_id
                map.add(line3d)
            Convert.save_map("Shape_Reviser", "schoolzone.osm", map, sub_dir="A3")

    class Correct_Roundabout():
        """
        회전교차로에 해당하는 A2 에 교차로속성 설정
        - LinkType ? -> 1
        """

        def execute(self):
            
            link_datas = Shape.get_shape_datas("A2")
            counter = Process_Counter(len(link_datas))

            for link_id, link in link_datas.items():
                if link.LinkType != 1:
                    from_node = Shape.get_shape("A1", link.FromNodeID)
                    to_node = Shape.get_shape("A1", link.ToNodeID)
                    if any([x.NodeType == 10 for x in [from_node, to_node]]):
                        link.replace(LinkType=1)
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[A2] Correct roundabout")
            counter.print_result("[A2] Correct roundabout")

    class Save_Intersection():

        def execute(self):

            map = Convert.create_map()
            for link_id, link in Shape.get_shape_datas("A2").items():
                if A2.check_intersection(link_id, is_post=False) == 1:
                    map.add(Convert.convert_to_lineString3d(link.points))
            Convert.save_map("Shape_Reviser", "intersection.osm", map, sub_dir="A2")

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.Correct_Oneway_Relation().execute()
        # cls.Break_Intersect_Relation().execute()
        # cls.Break_Close_Relation().execute()
        cls.Add_Missed_Node().execute()
        cls.Correct_Link_Pointer().execute()
        cls.Correct_Length().execute()
        cls.Deduplicate_points().execute()
        cls.Loosen_Dense_Points().execute()
        cls.Correct_Misdirected().execute()
        cls.Adjust_Gapped().execute()
        cls.Apply_Schoolzone().execute()
        # cls.Correct_Roundabout().execute()
        cls.Save_Intersection().execute()


class B2_Reviser():
    """
    B2 관련 정보 보완
    """

    class Break_StopLine():
        """
        정지선 A2 관계 제거
        """

        def execute(self):

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            for lane in lane_datas.values():
                if lane.Kind == 530:
                    lane.replace(L_LinkID=-1)
                    lane.replace(R_LinkID=-1)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[B2] Break stopLine")
            counter.print_result("[B2] Break stopLine")

    class Break_Intersection():
        """
        교차로 연관 B2 관계 제거
        """

        def execute(self):

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            for lane in lane_datas.values():
                for side_index in [0, 1]:
                    item = ["L_LinkID", "R_LinkID"][side_index]
                    link_id = getattr(lane, item)
                    if Shape.check_shape("A2", link_id):
                        if A2.check_intersection(link_id, is_post=False):
                            lane.replace(**{item : -1})
                            counter.add(item="fix")
                counter.add()
                counter.print_sequence("[B2] Break intersection")
            counter.print_result("[B2] Break intersection")

    class Remove_Isolated():
        """
        A2 관계가 없는 B2 제거
        """

        def execute(self):

            def save_target(target_datas):
                
                map = Convert.create_map()

                for lane_id, lane in target_datas.items():
                    line3d = Convert.convert_to_lineString3d(lane.points)
                    line3d.attributes["ID"] = str(lane_id)
                    map.add(line3d)

                Convert.save_map("Shape_Reviser", "isolated_lanes.osm", map, sub_dir="B2")

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            # 1. A2 관계가 존재하지 않는 B2 추출
            # - 정지선 (Kind = 530) 은 제외한다.
            target_datas = dict()            
            for lane_id, lane in lane_datas.items():
                if lane.Kind not in [530]:
                    if all([getattr(lane, item) == -1 for item in ["L_LinkID", "R_LinkID"]]):
                        target_datas[lane_id] = lane
                        counter.add(item="warn")
                counter.add()
                counter.print_sequence("[B2] Remove isolated")
            
            save_target(target_datas)

            # 2. B2 목록 갱신
            for lane_id, lane in target_datas.items():
                Shape.del_shape("B2", lane_id)
                counter.add(item="fix")
                counter.print_sequence("[B2] Remove isolated")
            counter.print_result("[B2] Remove isolated")

    class Deduplicate_points():
        """
        B2 중복좌표 제거
        """

        def execute(self):

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            for lane in lane_datas.values():
                # 1) 중복좌표제거 수행
                points = deduplicate_points(lane.points)
                # - 중복여부 검사
                if len(points) != len(lane.points):
                    # 2) 좌표목록 갱신
                    lane.replace(points=points)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[B2] Deduplicate points")
            counter.print_result("[B2] Deduplicate points")

    class Break_Intersect_Relation():
        """
        A2 - B2 교차관계 제거
        """

        def execute(self):

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            for lane in lane_datas.values():
                # 1) Left Link 검사
                left_link = Shape.get_shape("A2", lane.L_LinkID)
                if left_link != None:
                    # - A2 - B2 교차 시
                    if check_curve_intersect(lane.points, left_link.points):
                        # - B2 갱신
                        lane.replace(L_LinkID=-1)
                        counter.add(item="fix")
                        # log_print("[B2] Break intersection : {0}".format(lane.ID))

                # 2) Right Link 검사
                right_link = Shape.get_shape("A2", lane.R_LinkID)
                if right_link != None:
                    # - A2 - B2 교차 시
                    if check_curve_intersect(lane.points, right_link.points):
                        # - B2 갱신
                        lane.replace(R_LinkID=-1)
                        counter.add(item="fix")
                        # log_print("[B2] Break intersection : {0}".format(lane.ID))

                counter.add()
                counter.print_sequence("[B2] Break intersect relation")
            counter.print_result("[B2] Break intersect relation")

    class Correct_Reverse():
        """
        뒤집힌 B2 수정
        """

        def execute(self):

            def check_reverse(lane, link):
                prev_index = -1
                for point in lane.points:
                    curr_index = link.points.index(get_closest_point(point, link.points))
                    if prev_index != -1:
                        if curr_index < prev_index:
                            return True
                    prev_index = curr_index
                return False 

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            for lane in lane_datas.values():
                # 1) 좌/우 Link 추출
                side_links = [Shape.get_shape("A2", getattr(lane, item)) for item in ["L_LinkID", "R_LinkID"] if getattr(lane, item) != -1]
                for link in side_links:
                    # 2) 역전 검사
                    if check_reverse(lane, link):
                        # 3) 역전 수정
                        lane.replace(points=lane.points[::-1])
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[B2] Correct reverse")
            counter.print_result("[B2] Correct reverse")

    class Sample_Side():
        """
        잘못된 A2 방향 관계 제거
        - 추출 대상 : 잘못된 방향 설정 / 잘못된 관계 설정 (잘못된 A2 - B2 관계가 생성 당시에 설정)
        - map/Shape_Reviser/B2/sample_side.osm 에 저장된 A2 - B2 정보를 기반으로 shape 원본 파일의 오류를 수정
        """

        def execute(self):

            def check_side(lane, side_link, sample_count=3):

                if side_link == None:
                    return False

                # - 좌/우 방향 체크 (좌 : True)
                side_check = True if side_link.ID == lane.L_LinkID else False

                # - Lane 선분 추출
                for index, lane_seg in enumerate([lane.points[index:index+2] for index in range(len(lane.points)-1)]):
                    # - 표본에 해당하는 선분
                    if index in [count * int(len(lane.points) / sample_count) for count in range(sample_count)]:
                        # 1) 방향 비교
                        # - 표본 선분 중 올바른 A2 - B2 방향에 해당 시 True 반환
                        if check_is_left(lane_seg[0], lane_seg[-1], get_closest_point(get_mid(lane_seg[0], lane_seg[-1]), side_link.points)) == side_check:
                            return False
                # - 모든 표본이 올바르지 않은 방향인 경우 False 반환
                return True

            def save_sample(sample_datas):

                map = Convert.create_map()

                for (lane.points, sample_line) in sample_datas.values():
                    map.add(Convert.convert_to_lineString3d(lane.points))
                    map.add(Convert.convert_to_lineString3d(sample_line))

                Convert.save_map("Shape_Reviser", "sample_side.osm", map, sub_dir="B2")

            lane_datas = Shape.get_shape_datas("B2")
            counter = Process_Counter(len(lane_datas))

            sample_datas = dict()

            for lane in lane_datas.values():
                # 1) Lane 기준 선분 추출
                lane_seg = lane.points[:2]
                # 2) 좌/우 Link 추출
                for item in ["L_LinkID", "R_LinkID"]:
                    side_link = Shape.get_shape("A2", getattr(lane, item))
                    # - B2 - A2 간 방향이 반대인 경우
                    if check_side(lane, side_link):
                        # log_print("Test = {0} : {1}".format(lane.ID, item))
                        # 3) 추출 목록에 추가
                        sample_datas[lane.ID] = (lane.points, [get_mid(lane_seg[0], lane_seg[-1]), get_closest_point(get_mid(lane_seg[0], lane_seg[-1]), side_link.points)])
                        # 4) A2 - B2 관계 제거
                        lane.replace(**{item : -1})
                        counter.add(item="fix")
                counter.add()
                counter.print_sequence("[B2] Sample side")
            counter.print_result("[B2] Sample side")

            save_sample(sample_datas)

    # --------------------------------------------------

    @classmethod
    def execute(cls):

        # lane = Shape.get_shape("B2", "B219BS010971")
        # log_print("Test 1-1 = {0}".format([lane.L_LinkID, lane.R_LinkID]))
        # log_print("Test 1-2 = {0}".format(Shape.check_shape("B2", "B219BS010971")))
        
        cls.Break_StopLine().execute()
        # cls.Break_Intersection().execute()
        cls.Remove_Isolated().execute()
        cls.Deduplicate_points().execute()
        cls.Break_Intersect_Relation().execute()
        cls.Correct_Reverse().execute()
        cls.Sample_Side().execute()


class C1_Reviser():
    """
    C1 관련 정보 보완
    """

    class Remove_Undefined():
        """
        규정된 신호등 종류 이외 제거
        - 범위 : 1 ~ 15 
        """

        def execute(self):
            
            def check_range(light):
                if light.Type not in range(1, 16):
                    return False
                return True

            light_datas = Shape.get_shape_datas("C1")
            counter = Process_Counter(len(light_datas))

            for light_id, light in light_datas.items():
                if not check_range(light):
                    Shape.del_shape("C1", light_id)
                    counter.add(item="fix")
                counter.add()
                counter.print_sequence("[C1] Remove undefined")
            counter.print_result("[C1] Remove undefined")

    # --------------------------------------------------

    @classmethod
    def execute(cls):
        cls.Remove_Undefined().execute()


class Shape_Reviser(Module):

    def do_process(self, *args, **kwargs):
        
        A1_Reviser.execute()
        A2_Reviser.execute()
        B2_Reviser.execute()
        C1_Reviser.execute()

        return True
