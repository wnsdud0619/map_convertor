#!/usr/bin/python
# -*- coding: utf-8 -*-


from scripts.core.core_data import abstractclassmethod


class Altitude():

    @abstractclassmethod
    def get_map_points(cls, map_type):
        """
        각 유형에 해당하는 좌표 목록(list)을 반환
        - 1) Road
        - 2) Regulatory
        """
        pass


class Road():
    
    @abstractclassmethod
    def get_keys(cls):
        """
        Lanelet key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_side_key(cls, key, side_index):
        """
        특정 Lanelet(key) 의 좌/우 이웃에 해당하는 Lanelet key 를 반환
        - side_index : 0 == 좌
        - side_index : 1 == 우
        """
        pass

    @abstractclassmethod
    def get_bound(cls, key):
        """
        특정 Lanelet(key) 의 [leftBound, rightBound, centerLine] 목록 반환
        - 1) leftBound  : tuple(x, y, z) list
        - 2) rightBound : tuple(x, y, z) list
        - 3) centerLine : tuple(x, y, z) list [None 가능]
        """
        pass

    @abstractclassmethod
    def get_bound_attributes(cls, key, side_index):
        """
        특정 Lanelet(key) 의 좌/우 bound3d(LineString3d) 에 해당하는 속성(AttributeMap) 반환
        - side_index : 0 == 좌
        - side_index : 1 == 우
        """
        pass

    @abstractclassmethod
    def get_bound3d(cls, key):
        """
        특정 Lanelet(key) 의 [leftBound3d, rightBound3d, centerLine3d] 목록 반환
        - 1) leftBound3d  : LineString3d
        - 2) rightBound3d : LineString3d
        - 3) centerLine3d : LineString3d [None 가능]
        """
        pass

    @abstractclassmethod
    def set_bound3d(cls, key, bound3d):
        """
        특정 Lanelet(key) 의 bound3d = [leftBound3d, rightBound3d, centerLine3d] 목록 설정
        """
        pass

    @abstractclassmethod
    def get_lanelet_attributes(cls, key):
        """
        특정 Lanelet(key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_lanelet(cls, key):
        """
        key 에 해당하는 Lanelet 반환
        """
        pass

    @abstractclassmethod
    def set_lanelet(cls, key, lanelet):
        """
        특정 Lanelet 을 key 에 등록
        """
        pass
    
    # --------------------------------------------------

    @abstractclassmethod
    def get_row_lanelets(cls, key):
        """
        특정 lanelet(key) 의 전/후 Lanelet 목록 = [from, to] 반환 
        - from, to 는 None 가능
        """
        pass


class Crosswalk():
    
    @abstractclassmethod
    def get_keys(cls):
        """
        Crosswalk key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_bound(cls, key):
        """
        특정 Crosswalk(key) 에 해당하는 bound 목록 = [leftBound, rightBound] 반환
        - 1) leftBound  : tuple(x, y, z) list
        - 2) rightBound : tuple(x, y, z) list
        """
        pass

    @abstractclassmethod
    def get_bound_attributes(cls, key, side_index):
        """
        특정 Crosswalk(key) 의 좌/우 bound3d(LineString3d) 에 해당하는 속성(AttributeMap) 반환
        - side_index : 0 == 좌
        - side_index : 1 == 우
        """
        pass

    @abstractclassmethod
    def get_lanelet_attributes(cls, key):
        """
        특정 Crosswalk(key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass    

    @abstractclassmethod
    def get_lanelet(cls, key):
        """
        key 에 해당하는 Crosswalk 반환
        """
        pass

    @abstractclassmethod
    def set_lanelet(cls, key, lanelet):
        """
        특정 Crosswalk 을 key 에 등록
        """
        pass
    

class Light():

    @abstractclassmethod
    def get_keys(cls):
        """
        Regulatory key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_refer_keys(cls, key):
        """
        특정 Regulatory(key) 의 refer key 목록 반환 
        """
        pass           

    @abstractclassmethod
    def get_refer(cls, refer_key):
        """
        특정 refer(refer_key) 에 해당하는 좌표목록 반환
        - 좌표목록 : [(x1, y1, z1), (x2, y2, z2)] (신호등을 마주보는 기준에서 좌 -> 우)
        """
        pass

    @abstractclassmethod
    def get_refer_offset(cls, refer_key):
        """
        refer 의 XY 좌표에 해당하는 Altitude 고도값에 적용될 Z offset
        - Lanelet2 형식은 refer 의 Z 값이 신호등의 바닥이 기준
        - 변환 전 형식의 refer 의 Z 값이 신호등의 바닥이 기준이 아닌 경우 그 차이만큼 refer_offset 반환
        - ex) SHP 형식은 신호등의 중심이 기준 => (바닥 - 중심) Z 값을 반환 
        """
        pass

    @abstractclassmethod
    def get_refer_attributes(cls, refer_key):
        """
        특정 refer3d(refer_key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_bulb(cls, refer_key):
        """
        특정 refer(refer_key) 에 대응하는 bulb 좌표목록 반환
        - 좌표목록 : tuple(x, y, z) list 
        - 좌표목록의 각 좌표는 전구의 위치를 의미
        - 좌표목록의 개수는 전구의 개수와 동일
        """
        pass

    @abstractclassmethod
    def get_bulb_offset(cls, refer_key, index):
        """
        bulb 의 XY 좌표에 해당하는 Altitude 고도값에 적용될 Z offset
        - Lanelet2 형식은 bulb 의 Z 값이 전구의 중심 기준
        - 변환 전 형식의 bulb 의 Z 값이 전구의 중심 기준이 아닌 경우 그 차이만큼 bulb_offset 반환
        """
        pass

    @abstractclassmethod
    def get_bulb_attributes(cls, refer_key):
        """
        특정 bulb3d(refer_key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_bulb_color(cls, refer_key, index):
        """
        특정 bulb3d 를 구성하는 N 번째 Point3d(전구)의 전구 색깔 반환
        """
        pass

    @abstractclassmethod
    def get_bulb_arrow(cls, refer_key, index):
        """
        특정 bulb3d 를 구성하는 N 번째 Point3d(전구)의 화살표 방향 반환
        - N 번째 Point3d 가 화살표가 아닌경우 None 반환
        """
        pass

    @abstractclassmethod
    def get_refer_origin(cls, refer_key):
        """
        신호등(refer3d, bulb3d)을 구성하는 모든 Point3d 의 기준에 해당하는 기준 좌표 반환
        - 기준 좌표 : tuple(x, y, z)
        - 모든 Point3d 의 고도값은 기준 좌표의 고도값에 offset 을 적용해 추출
        """
        pass

    @abstractclassmethod
    def get_stopLine_key(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 StopLine key 반환
        """
        pass

    @abstractclassmethod
    def get_stopLine(cls, stopLine_key):
        """
        특정 StopLine(stopLine_key) 반환
        """
        pass

    @abstractclassmethod
    def get_stopLine_attributes(cls, stopLine_key):
        """
        특정 StopLine(stopLine_key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_lanelet_keys(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 Lanelet key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_regulatory_attributes(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    # --------------------------------------------------

    @abstractclassmethod
    def get_refer3d(cls, refer_key):
        """
        refer(refer_key) 에 해당하는 refer3d 반환
        """
        pass

    @abstractclassmethod
    def set_refer3d(cls, refer_key, refer3d):
        """
        refer3d 를 refer_key 에 등록
        """
        pass

    @abstractclassmethod
    def get_bulb3d(cls, refer_key):
        """
        refer(refer_key) 에 해당하는 bulb3d 반환
        """
        pass

    @abstractclassmethod
    def set_bulb3d(cls, refer_key, bulb3d):
        """
        bulb3d 를 refer_key 에 등록
        """
        pass

    @abstractclassmethod
    def get_stopLine3d(cls, stopLine_key):
        """
        stopLine(stopLine_key) 에 해당하는 stopLine3d 반환
        """
        pass

    @abstractclassmethod
    def set_stopLine3d(cls, stopLine_key, stopLine3d):
        """
        stopLine3d 를 stopLine_key 에 등록
        """
        pass

    @abstractclassmethod
    def get_regulatory(cls, key):
        """
        key 에 해당하는 Regulatory 반환
        """
        pass

    @abstractclassmethod
    def set_regulatory(cls, key, regulatory):
        """
        Regulatory 를 key 에 등록
        """
        pass

    
class Sign():

    @abstractclassmethod
    def get_keys(cls):
        """
        Regulatory key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_refer_keys(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 refer key 목록 반환
        """
        pass      

    @abstractclassmethod
    def get_refer(cls, refer_key):
        """
        refer_key 에 해당하는 좌표목록 반환
        - 좌표목록 : [(x1, y1, z1), (x2, y2, z2)] (신호표지를 마주보는 기준에서 좌 -> 우)
        """
        pass

    @abstractclassmethod
    def get_refer_offset(cls, refer_key):
        """
        refer 의 XY 좌표에 해당하는 Altitude 고도값에 적용될 Z offset
        - Lanelet2 형식은 refer 의 Z 값이 신호표지의 바닥 기준
        - 변환 전 형식의 refer 의 Z 값이 신호표지의 바닥 기준이 아닌 경우 그 차이만큼 refer_offset 반환
        - ex) SHP 형식은 신호표지의 중심이 기준 => (바닥 - 중심) Z 값을 반환 
        """
        pass

    @abstractclassmethod
    def get_refer_attributes(cls, refer_key):
        """
        특정 refer3d(refer_key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_refer_origin(cls, refer_key):
        """
        신호표지(refer3d)를 구성하는 모든 Point3d 의 기준에 해당하는 기준 좌표 반환
        - 기준 좌표 : tuple(x, y, z)
        - 모든 Point3d 의 고도값은 기준 좌표의 고도값에 offset 을 적용해 추출
        """
        pass

    @abstractclassmethod
    def get_stopLine_key(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 StopLine key 반환
        """
        pass

    @abstractclassmethod
    def get_stopLine(cls, stopLine_key):
        """
        특정 StopLine(stopLine_key) 반환
        """
        pass

    @abstractclassmethod
    def get_stopLine_attributes(cls, stopLine_key):
        """
        특정 StopLine(stopLine_key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    @abstractclassmethod
    def get_lanelet_keys(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 Lanelet key 목록 반환
        """
        pass

    @abstractclassmethod
    def get_nation_code(cls, key):
        """
        국가 코드 반환
        - ex : 한국 = kr
        """
        pass    

    @abstractclassmethod
    def get_regulatory_attributes(cls, key):
        """
        특정 Regulatory(key) 에 해당하는 속성(AttributeMap) 반환
        """
        pass

    # --------------------------------------------------

    @abstractclassmethod
    def get_refer3d(cls, refer_key):
        """
        refer(refer_key) 에 해당하는 refer3d 반환
        """
        pass

    @abstractclassmethod
    def set_refer3d(cls, refer_key, refer3d):
        """
        refer3d 를 refer_key 에 등록
        """
        pass

    @abstractclassmethod
    def get_stopLine3d(cls, stopLine_key):
        """
        stopLine(stopLine_key) 에 해당하는 stopLine3d 반환
        """
        pass

    @abstractclassmethod
    def set_stopLine3d(cls, stopLine_key, stopLine3d):
        """
        stopLine3d 를 stopLine_key 에 등록
        """
        pass

    @abstractclassmethod
    def get_regulatory(cls, key):
        """
        key 에 해당하는 Regulatory 반환
        """
        pass

    @abstractclassmethod
    def set_regulatory(cls, key, regulatory):
        """
        Regulatory 를 key 에 등록
        """
        pass
