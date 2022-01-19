#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
from functools import (reduce)
from scipy.spatial import (ConvexHull)
from scipy.ndimage import (gaussian_filter1d)
import matplotlib.pyplot as plt

from utm import (from_latlon)

from print_functions import (
    log_print,
    warning_print,
)

np.seterr(invalid='raise')

# [좌표계] --------------------------------------------------

def convert_latlon_to_utm(points):
    """
    위경도 좌표목록(points)을 UTM 형식으로 변환 후 반환
    """
    new_points = []
    for point in points:
        x, y, zone_number, zone_letter = from_latlon(point[0], point[1])
        new_point = (x, y) if len(point) <= 2 else (x, y, point[2])
        new_point = tuple([round(x, 5) for x in new_point])
        new_points.append(new_point)
    return new_points

# [거리] --------------------------------------------------

def calc_distance(p1, p2, is_alt=False):
    """
    p1, p2 사이의 거리 반환
    1) x, y 평면 거리 반환
    2) x, y, z 3차원 거리 반환
    """
    
    x_diff = (p1[0] - p2[0])**2 
    y_diff = (p1[1] - p2[1])**2
    
    if not is_alt:
        return math.sqrt(x_diff + y_diff)
    
    z_diff = (p1[2] - p2[2])**2
    return math.sqrt(x_diff + y_diff + z_diff)

def calc_length(points):
    """
    좌표목록(points)의 길이 총합을 반환
    """

    length = 0.0
    for seg in [points[index:index+2] for index in range(len(points)-1)]:
        length += calc_distance(seg[0], seg[-1])
    return length

def calc_distance_from_line(line, point):
    """
    2차원 평면(x, y) 상에서 선(line) - 점(point) 간 수직거리를 반환
    """

    p1 = np.array([line[0][0], line[0][1]])
    p2 = np.array([line[1][0], line[1][1]])
    p3 = np.array([point[0], point[1]])
    
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def calc_distance_from_point(origin, target, slope=None, start=None, end=None, is_sign=False):
    """
    원점(origin)에서 좌표(target) 까지의 수직 거리를 반환
    - 수직 거리 : 기울기(slope)에 수직이며 원점을 지나는 선으로부터 좌표까지의 거리
    """

    # 1. 수직선 추출
    ortho_line = get_ortho_line(origin, start=start, end=end) if slope == None else get_ortho_line(origin, slope=slope)
    # 2. 거리 계산
    distance = calc_distance_from_line(ortho_line, target)

    # - 거리의 부호 구분이 필요한 경우 (기울기에 기반)    
    if is_sign:
        sign = -1 if check_is_left(ortho_line[0], ortho_line[-1], target) else 1
        return distance * sign

    return distance

def calc_distance_from_point_test():
    
    origin = (0,0,0)
    
    targets = [
        (0, 1, 0),
        (1, 0, 0),
        (0, -1, 0),
        (-1, 0, 0)
    ]

    end = (1, 0, 0)
    
    predicts = [
        0.0,
        1.0,
        0.0,
        -1.0
    ]

    for index in range(len(targets)):
        distance = calc_distance_from_point(origin, targets[index], start=origin, end=end, is_sign=True)
        if distance != predicts[index]:
            log_print("{0} : {1} / {2}".format(index, distance, predicts[index]))
            break
    else:
        log_print("Test Complete [OK]")        

def calc_z(p1, p2, target):

    # x, y 값이 동일한 경우 z 값을 구할 수 없다.
    if all([p1[i] == p2[i] for i in range(2)]):
        return None

    # 1. xz_slope 추출
    xz_slope = calc_slope(p1, p2, is_alt=True)

    if xz_slope == float("inf"):
        yz_slope = calc_slope((p1[1], p1[2]), (p2[1], p2[2]))
        yz_bias = p1[2] - yz_slope * p1[1]
        target_z = yz_slope * target[1] + yz_bias
    elif xz_slope == 0:
        # - 기울기가 수직인 경우 z 값 추측이 불가능
        target_z = p1[2]
    else:
        # 2. xz_bias 추출
        xz_bias = p1[2] - xz_slope * p1[0]
        # 3. 목표 z 값 계산
        target_z = xz_slope * target[0] + xz_bias

    return target_z

def calc_z_test():

    p1 = (0,0,0)
    p2 = (1,1,1)
    
    targets = [
        (-1,-1),
        (2,2),
        (0,0)
    ]
    
    predicts = [
        -1, 2, 0
    ]

    for index in range(len(targets)):
        result = calc_z(p1, p2, targets[index])
        log_print("Result : {0} / {1}".format(result, predicts[index]))
        if result != predicts[index]:
            break
    else:
        log_print("Test Complete [OK]")

def calc_length_on_points(points, target):
    """
    좌표목록(points) 시작 좌표에서 좌표목록 상에 존재하는 특정 좌표(target) 까지의 거리 합 반환
    """

    length = 0.0
    for segment in [points[index:index+2] for index in range(len(points)-1)]:
        if check_point_on_line(segment, target):
            return length + calc_distance(segment[0], target)
        length += calc_distance(segment[0], segment[-1])

    return None

def calc_length_on_points_test():

    points_list = [
        [
            (0,0,0),
            (10,0,0)
        ],
        [
            (0,0,0),
            (5,5,0),
            (10,5,0),
            (15,0,0),
            (20,0,0)
        ]
    ]

    targets = [
        (5,0,0),
        (12.5, 2.5, 0)
    ] 

    predicts = [
        5.0,
        calc_distance((0,0,0), (5,5,0)) * 1.5 + 5.0
    ]

    for index in range(len(points_list)):
        result = calc_length_on_points(points_list[index], targets[index])
        if result != predicts[index]:
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def calc_z_from_plane(plane, x, y):
    """
    좌표 3개로 구성된 평면(plane)에서 (x, y) 좌표에 해당하는 z 값을 반환 
    """

    p1 = plane[0]
    p2 = plane[1]
    p3 = plane[2]

    p1 = list([float(a) for a in p1])
    p2 = list([float(a) for a in p2])
    p3 = list([float(a) for a in p3])
    x = float(x)
    y = float(y)

    A = p1[1] * (p2[2] - p3[2]) + p2[1] * (p3[2]- p1[2]) + p3[1] * (p1[2] - p2[2])
    B = p1[2] * (p2[0] - p3[0]) + p2[2] * (p3[0] - p1[0]) + p3[2] * (p1[0] - p2[0])
    C = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    D = -(p1[0] * (p2[1] * p3[2] - p3[1] * p2[2]) + p2[0] * (p3[1] * p1[2] - p1[1] * p3[2]) + p3[0] * (p1[1] * p2[2] - p2[1] * p1[2]))

    try:
        z = (-(A * x + B * y + D)) / C
    except ZeroDivisionError:
        z = None

    return z

def calc_z_from_plane_test():

    plane = [
        (0,0,0),
        (10,10,5),
        (10,0,5)
    ]

    points = [
        (0,0),
        (5,5),
        (5,0),
        (10,5),
        (20,10),
        (-10,10),
    ]

    predicts = [
        0,
        2.5,
        2.5,
        5,
        10,
        -5
    ]

    for index in range(len(points)):
        result = calc_z_from_plane(plane, points[index][0], points[index][1])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def check_colinear(p1, p2, p3):

    a = p1[0] * (p2[1] - p3[1]) 
    b = p2[0] * (p3[1] - p1[1])
    c = p3[0] * (p1[1] - p2[1])

    check = (a + b + c) == 0.0
    
    return check

# [각도] --------------------------------------------------

def calc_degree(p1, p2):
    """
    원점(0,0) 기준 p1, p2 벡터 사이각을 반환 (x,y 평면 기준)
    """
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    slope = calc_slope(p1, p2)
    degree = math.atan(slope) * (180.0/math.pi)

    if dx < 0.0: 
        degree += 180.0
    else:
        if dy < 0.0:
            if dx == 0.0:
                degree += 180.0
            else:
                degree += 360.0

    return degree

def normalize_deg(degree):
    """
    각도를 0 ~ 360 도 범위로 변환
    """
    degree = degree % 360.0
    degree = degree if degree > 0 else degree + 360
    return degree

def calc_average_degree(degrees):
    """
    각도 평균 반환
    """

    # 1. 각도 범위 조정
    degrees = [normalize_deg(x) for x in degrees]
    # 2. 벡터 변환
    vectors = [convert_to_vector(x) for x in degrees]
    # 3. 평균 벡터 추출
    average_vector = reduce(lambda x, y : tuple([x[index] + y[index] for index in range(3)]), vectors, (0,0,0))
    # 4. 평균벡터 각도 추출
    average_degree = calc_degree((0,0,0), average_vector) % 360.0

    return average_degree

def calc_average_degree_test():

    degree_lists = [
        [-1, 1],
        [1, 359],
        [89, 91],
        [89, -269], 
        [179, 181],
        [179, -179],
        [269, -89],
        [269, 271],
        [30, 60],
        [90, 180],
        [180, 270],
        [270, 360]
    ]

    predicts = [
        0, 0, 
        90, 90,
        180, 180, 
        270, 270, 
        45, 135, 225, 315
    ]

    for index in range(len(degree_lists)):
        result = calc_average_degree(degree_lists[index])
        if result != predicts[index]:
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def convert_to_vector(degree, line=None):
    """
    각도를 벡터로 변환
    - 벡터의 규모(magnitude)는 1
    """

    degree = (degree % 360)
    degree = degree + 360 if degree < 0 else degree

    if line != None:
        degree = calc_degree(line[0], line[1])

    if degree == 0:
        return (1.0, 0, 0)
    elif degree == 90:
        return (0, 1.0, 0)
    elif degree == 180:
        return (-1.0, 0, 0)
    elif degree == 270:
        return (0, -1.0, 0)

    slope = convert_to_slope(degree)
    sign = -1.0 if 90 < degree < 270 else 1.0
    x = sign / math.sqrt(slope**2 + 1)

    return (x, x * slope, 0)

def convert_to_vector_test():

    degrees = [
        0,
        90,
        180,
        270,
        360,
        -0,
        -270,
        -180,
        -90,
        45,
        -315,
        135,
        -225,
        225,
        -135,
        315,
        -45
    ]

    predicts = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0),
        (1,0,0),
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0),
        (math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0),
        (math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0),
        (-math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0),
        (-math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0),
        (-math.sqrt(2)/2.0, -math.sqrt(2)/2.0, 0),
        (-math.sqrt(2)/2.0, -math.sqrt(2)/2.0, 0),
        (math.sqrt(2)/2.0, -math.sqrt(2)/2.0, 0),
        (math.sqrt(2)/2.0, -math.sqrt(2)/2.0, 0),
    ]

    for index in range(len(degrees)):
        result = convert_to_vector(degrees[index])
        if not all([round(result[i], 5) == round(predicts[index][i], 5) for i in range(3)]):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def calc_vector_theta(v1, v2):
    """
    2개의 벡터(v1, v2)의 사잇각 반환
    - 항상 양수값 반환
    """

    mul = (v1[0] * v2[0] + v1[1] * v2[1])
    val = (calc_distance((0,0,0), v1) * calc_distance((0,0,0), v2))

    # - v1 * v2 / [v1] * [v2] 값이 1인 경우 => cos theta 가 1이란 뜻이므로 theta = 0
    if round(mul, 5) == round(val, 5):
        theta = 0.0
    else:
        theta = math.degrees(math.acos(mul / val))

    return theta

def calc_vector_theta_test():

    vector_pairs = [
        [
            (1,0,0),
            (1,0,0)
        ],
        [
            (1,0,0),
            (1,1,0)
        ],
        [
            (1,0,0),
            (0,1,0)
        ],
        [
            (1,0,0),
            (-1,1,0)
        ],
                [
            (1,0,0),
            (-1,0,0)
        ],
                [
            (1,0,0),
            (-1,-1,)
        ],
                [
            (1,0,0),
            (0,-1,0)
        ],
        [
            (1,0,0),
            (1,-1,0)
        ],
    ]

    predicts = [
        0.0, 45.0, 90.0, 135.0, 180.0, 135.0, 90.0, 45.0 
    ]

    for index in range(len(vector_pairs)):
        result = calc_vector_theta(vector_pairs[index][0], vector_pairs[index][1])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

    # todo : 테스트 완료하기

def calc_degree_diff(d1, d2):
    """
    각도 2개의 차이를 반환
    - 각도의 최소 차이는 0도
    - 각도의 최대 차이는 180도
    """

    v1, v2 = [convert_to_vector(d) for d in [d1, d2]]

    theta = calc_vector_theta(v1, v2)

    return theta

def calc_degree_diff_test():
    
    degree_pairs = [
        [0, 1],
        [0, -1],
        [0,361],
        [0,359],
        [0, 179],
        [0, -179],
        [0, -181],
        [0, 181],
    ]

    predicts = [
        1,
        1,
        1,
        1,
        179,
        179, 
        179,
        179
    ]

    for index in range(len(degree_pairs)):
        result = calc_degree_diff(degree_pairs[index][0], degree_pairs[index][1])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def convert_to_degree(slope, vector=None):
    """
    기울기를 각도로 변환 후 반환
    """

    if vector != None:
        degree = calc_degree((0,0,0), vector)
    else:
        degree = math.degrees(math.atan(slope))

    return degree

def convert_to_degree_test():
    
    slopes = [
        0,
        float("inf"),
        .5,
        1,
        2,
        -.5,
        -1,
        -2
    ]

    predicts = [
        0,
        90,
        calc_vector_theta((1,0,0), (1, .5, 0)),
        calc_vector_theta((1,0,0), (1, 1, 0)),
        calc_vector_theta((1,0,0), (1, 2, 0)),
        -calc_vector_theta((1,0,0), (1, -.5, 0)),
        -calc_vector_theta((1,0,0), (1, -1, 0)),
        -calc_vector_theta((1,0,0), (1, -2, 0)),
    ]

    for index in range(len(slopes)):
        result = convert_to_degree(slopes[index])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test 1 Complete [OK]")

    vectors = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0),
        (1,1,0),
        (-1,1,0),
        (-1,-1,0),
        (1,-1,0),
    ]

    predicts = [
        0,
        90,
        180,
        270,
        45,
        135,
        225,
        315
    ]

    for index in range(len(vectors)):
        result = convert_to_degree(None, vector=vectors[index])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test 2 Complete [OK]")

def convert_to_slope(degree):
    return math.tan(math.radians(degree))

def calc_slope(p1, p2, is_alt=False):
    """
    좌표 2개에 기반한 기울기 반환
    is_alt == False : (x, y) 평면 기울기
    is_alt == True : (x, z) 평면 기울기 
    """

    coord_index = 1 if not is_alt else 2    

    if p1[0] == p2[0]:
        return float("inf")
    if p1[coord_index] == p2[coord_index]:
        return 0

    x_diff = float(p1[0]) - float(p2[0])
    coord_diff = float(p1[coord_index]) - float(p2[coord_index])
    
    return coord_diff / x_diff

def calc_curve_slope(points):
    """
    좌표목록의 평균 기울기 반환
    - 1차함수 기울기 반환 
    """

    # 좌표 개수가 2개 미만
    if len(points) < 2:
        log_print("Cannot calculate 1 point slope")
        sys.exit()
    # 좌표 개수가 2개
    elif len(points) == 2:
        slope = calc_slope(points[0], points[1])
    # 모든 x 값이 동일한 경우 (y 축)
    elif all([x[0] == points[0][0] for x in points[1:]]):
        slope = float("inf")
    # 모든 y 값이 동일한 경우 (x 축)
    elif all([x[1] == points[0][1] for x in points[1:]]):
        slope = 0
    # 일반적인 상황
    else:
        x, y = zip(*[(point[0], point[1]) for point in points])
        slope = np.polyfit(x, y, 1)[0]

    return slope

def calc_curve_degree(points):
    """
    좌표목록의 평균 각도 반환
    - calc_curve_slope (np.polyfit) 에 기반
    """

    if len(points) < 2:
        log_print("Cannot calculate 1 point degree")
        sys.exit()

    curve_slope = calc_curve_slope(points)

    return convert_to_degree(curve_slope)

def calc_curve_diff(points, compare_points=None):
    """
    좌표목록 간 시작 - 종료 각도차 반환
    - 1) compare_points == None : points 시작 - 종료 각도차
    - 2) compare_points != None : points - compare_points 각도차
    """

    #* (210513 수정)

    if compare_points != None:
        v1 = normalize(tuple([points[1][i] - points[0][i] for i in range(3)]))
        v2 = normalize(tuple([compare_points[1][i] - compare_points[0][i] for i in range(3)]))
    else:
        v1 = normalize(tuple([points[1][i] - points[0][i] for i in range(3)]))
        v2 = normalize(tuple([points[-1][i] - points[-2][i] for i in range(3)]))

    theta = calc_vector_theta(v1, v2)

    return theta

    #* (~210513)

    # s_slope = calc_curve_slope([points[0], points[1]])
    # e_slope = calc_curve_slope([points[-2], points[-1]])

    # s_degree = convert_to_degree(s_slope)
    # e_degree = convert_to_degree(e_slope)

    # return calc_degree_diff(s_degree, e_degree)

def calc_curve_diff_test():
    
    points_list = [
        [
            (0,0,0), (1,0,0),
            (2,0,0), (3,0,0)            
        ],
        [
            (0,0,0), (0,1,0),
            (0,2,0), (0,3,0)            
        ],
        [
            (0,0,0), (-1,0,0),
            (-2,0,0), (-3,0,0)            
        ],
        [
            (0,0,0), (0,-1,0),
            (0,-2,0), (0,-3,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (3,1,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (2,1,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (1,1,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (1,0,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (1,-1,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (2,-1,0)            
        ],
        [
            (0,0,0), (1,0,0),
            (2,0,0), (3,-1,0)            
        ],
    ]

    predicts = [
        0,
        0,
        0,
        0,
        45,
        90,
        135,
        180,
        135,
        90,
        45
    ]

    for index in range(len(points_list)):
        result = calc_curve_diff(points_list[index])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

    for index in range(len(points_list)):
        result = calc_curve_diff(points_list[index][:2], compare_points=points_list[index][-2:])
        if round(result, 5) != round(predicts[index], 5):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def calc_ortho_slope(p1, p2):
    
    slope = calc_slope(p1, p2)
    
    if slope == 0:
        return float("inf")
    elif slope == float("inf"):
        return 0
    
    return -1.0 / slope

def rotate_point(point, origin=None, deg=None, angle=None):
    
    def rotate(np_point, angle):

        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])

        np_point = np_point.dot(rotation)

        return np_point

    if origin == None:
        origin = (0,0,0)

    if deg != None:
        angle = math.radians(deg)

    _point = (point[0], point[1], 0)

    # 1. 평행이동
    start = (origin[0], origin[1], 0)
    end = (0,0,0)
    _point = move_point(_point, None, start=start, end=end)

    # 2. 회전
    point2d = (_point[0], _point[1])
    np_point = np.array(point2d)
    np_point = rotate(np_point, angle)
    _point = (np_point[0], np_point[1], 0)

    # 3. 평행이동
    start = (0,0,0)
    end = (origin[0], origin[1], 0)
    _point = move_point(_point, None, start=start, end=end)
    
    point = (_point[0], _point[1], point[2])

    return point

def rotate_point_test():

    def check_same(p1, p2):
        
        for index in range(3):
            v1 = round(p1[index], 5)
            v2 = round(p2[index], 5)
            if v1 != v2:
                return False

        return True

    point = (1,0,0)

    degs = [0, -45, -90, -180, -360]

    origin = (0,0,0)

    predicts = [
        (1,0,0),
        (math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0),
        (0,1,0),
        (-1,0,0),
        (1,0,0)
    ]

    for index in range(len(degs)):
        result = rotate_point(point, origin=origin, deg=degs[index])
        if not check_same(result, predicts[index]):
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predicts : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 1 Complete [OK]")

    origin = (1,0,0)

    for index in range(len(degs)):
        result = rotate_point(point, origin=origin, deg=degs[index])
        if not check_same(result, origin):
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predicts : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 2 Complete [OK]")

    origin = (2,0,0)

    predicts = [
        (1,0,0),
        (2.0 - math.sqrt(2)/2.0, -math.sqrt(2)/2.0, 0),
        (2,-1,0),
        (3,0,0),
        (1,0,0)
    ]

    for index in range(len(degs)):
        result = rotate_point(point, origin=origin, deg=degs[index])
        if not check_same(result, predicts[index]):
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predicts : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 3 Complete [OK]")

def rotate_seg(seg, origin=None, deg=None, angle=None):
    """
    회전방향은 시계방향 (사분면 각도와 반대방향)
    """

    if origin == None:
        origin = get_mid(seg[0], seg[-1])

    if deg != None: 
        angle = math.radians(deg)

    seg = [rotate_point(p, origin=origin, angle=angle) for p in seg]

    return seg
    
def rotate_seg_test():

    def check_same(seg1, seg2):
        
        for i in range(len(seg1)):
            p1 = seg1[i]
            p2 = seg2[i]
            for j in range(3):
                v1 = round(p1[j], 5)
                v2 = round(p2[j], 5)
                if v1 != v2:
                    return False

        return True

    seg = [(0,0,0), (1,0,0)]

    degs = [0, -45, -90, -180, -360]

    origin = (0,0,0)

    predicts = [
        [(0,0,0), (1,0,0)],
        [(0,0,0), (math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0)],
        [(0,0,0), (0,1,0)],
        [(0,0,0), (-1,0,0)],
        [(0,0,0), (1,0,0)],
    ]

    for index in range(len(degs)):
        result = rotate_seg(seg, deg=degs[index])
        if not check_same(result, predicts[index]):
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predicts : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 1 Complete [OK]")

    seg = [(1,0,0), (2,0,0)]

    origin = (1,0,0)

    predicts = [
        [(1,0,0), (2,0,0)],
        [(1,0,0), (1.0 + math.sqrt(2)/2.0, math.sqrt(2)/2.0, 0)],
        [(1,0,0), (1,1,0)],
        [(1,0,0), (0,0,0)],
        [(1,0,0), (2,0,0)],
    ]

    for index in range(len(degs)):
        result = rotate_seg(seg, origin=origin, deg=degs[index])
        if not check_same(result, predicts[index]):
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predicts : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 2 Complete [OK]")

# [필터링] --------------------------------------------------

def deduplicate_points(points):
    """
    중복 좌표 제거 후 반환
    """

    new_points = points[:1]
    for point in points[1:]:
        dist = calc_distance(new_points[-1], point)
        if dist > 0.001:
            new_points.append(point)

    return new_points

def remove_close(points, close_dist):
    """
    좌표 중 기준거리(close_dist) 이하로 근접한 2개의 인접 좌표쌍이 있는 경우 늦은 순번의 좌표를 제거한다.
    - 시작/종료 좌표는 제거대상에서 제외한다.
    - N - 2 까지는 정순으로 제거
    - N - 1 은 역순으로 제거 (빠른 순번의 좌표를 제거) 
    """

    # 1. N-2 까지 정순으로 제거 (뒤의 좌표를 제거)
    new_points = points[:1]
    for index in range(1, len(points) - 1):
        prev_point = new_points[-1]
        next_point = points[index]
        if calc_distance(prev_point, next_point) > close_dist:
            new_points.append(next_point)
    
    # 2. N-1 은 역순으로 제거 (앞의 좌표를 제거)
    if calc_distance(new_points[-1], points[-1]) <= close_dist:
        new_points.pop(-1)
    new_points.append(points[-1])

    return new_points

def correct_misdirected(points):
    """
    좌표목록(points)의 진행방향에 반대되는 좌표를 제거
    """

    new_points = points[:1]

    for index in range(1, len(points)-1):
        if calc_distance(new_points[-1], points[index]) < calc_distance(new_points[-1], points[index+1]):
            new_points.append(points[index])
    new_points.append(points[-1])
    
    return new_points

    remove_list = []
    prev_point = points[0]
    index = 1
    while index <= len(points) - 2:
        # 다음 좌표가 진행하지 않고 역행한다면
        if calc_distance(prev_point, points[index]) > calc_distance(prev_point, points[index+1]):
            remove_list.append(points[index])
            index += 1
        index += 1

    if len(remove_list) > 0:
        points = [x for x in points if x not in remove_list]

    return points

# [기하] --------------------------------------------------

def check_same(p1, p2, floating=5):
    """
    좌표 2개의 좌표값이 소수점 N 자리까지 동일한 경우 True 반환
    """
    return (
        round(p1[0], floating) == round(p2[0], floating) and 
        round(p1[1], floating) == round(p2[1], floating)
        )

def select_straight(lines, is_index=False):
    
    min_diff = float("inf")
    min_index = -1

    for index, line in enumerate(lines):
        diff = calc_curve_diff(line)
        if diff < min_diff:
            min_diff = diff
            min_index = index

    result = min_index if is_index else lines[min_index]

    return result

def select_straight_test():
    
    lines_list = [
        [
            [(0,0,0), (10,0,0)],
            [(0,0,0), (5,0,0), (10,1,0)]
        ],
        [
            [(0,0,0), (10,0,0)],
            [(0,1,0), (10,1,0)],
        ],
    ]

    predicts = [0, 0]

    for index, lines in enumerate(lines_list):
        result = select_straight(lines, is_index=True)
        if result != predicts[index]:
            log_print("Index = {0}".format(index))
            log_print("Predice = {0}".format(predicts[index]))
            log_print("Result = {0}".format(result))
    else:
        log_print("Test Complete [OK]")

def get_closest_point(origin, points):
    """
    좌표목록(points) 좌표 중에서 기준점(origin)에 가장 가까운 좌표 1개를 반환
    """

    # - 좌표목록 구성 오류 시
    if len(points) < 1:
        log_print("Cannot calculate empty points")
        return None

    # - 좌표목록 길이가 1인 경우
    if len(points) < 2:
        return points[0]

    # - 좌표목록 길이가 2 이상인 경우 : 최단거리 좌표 추출 및 반환
    return sorted(points, key=lambda x : calc_distance(origin, x))[0]

def get_closest_segment(origin, points, is_index=False):
    """
    좌표목록(points)에서 원점(origin)에 가장 가까운 선분(연속한 두 점)을 순서에 맞게 반환
    - 좌표목록(points)의 목차 기준으로 빠른것이 선분의 첫번째 좌표가 된다.
    """

    # - 좌표목록 개수가 1개 이하인 경우 
    if len(points) < 2:
        log_print("Need at least 2 points")
        return None

    # - 좌표목록 개수가 2개 이하인 경우 
    if len(points) < 3:
        # - 목차 순 = 그대로 반환
        return points 

    # 1. 최단거리 좌표 추출 (1개)
    first = get_closest_point(origin, points)

    # 2. 최단거리 인접 좌표 추출
    # - 최단거리 좌표가 시작/끝 좌표인 경우 : 2, N-1 좌표를 추출
    if first in [points[0], points[-1]]:
        second = points[1] if first == points[0] else points[-2] 
    # - 최단거리 좌표가 시작/끝 좌표가 아닌 경우 : 보다 기준점(origin)에 더 가까운 좌표를 추출
    else:
        if is_index:
            second = points[points.index(first) + 1]
        else:
            # - 전/후 좌표 중 기준점에 더 가까운 좌표 추출
            second = sorted([points[points.index(first) - 1], points[points.index(first) + 1]], key=lambda x : calc_distance(origin, x))[0]

    # 3. 기존 좌표목록 목차 순으로 정렬
    segment = sorted([first, second], key=lambda x : points.index(x))
    
    return segment

def move_point(origin, slope, start=None, end=None, distance=None, is_alt=False):
    
    def __dy(distance, m, y_sign):
        if m == 0:
            return 0.0
        elif m == float("inf"):
            return distance * y_sign

        distance = float(distance)
        m = float(m)
        return m * __dx(distance, m)

    def __dx(distance, m):
        if m == 0:
            return distance
        elif m == float("inf"):
            return 0.0

        distance = float(distance)
        m = float(m)
        return distance / math.sqrt((m**2 + 1))

    x_sign = 1
    y_sign = 1

    if slope == None:
        slope = calc_slope(start, end)
        x_sign, y_sign = [np.sign(end[index] - start[index]) for index in [0, 1]]
        x_sign, y_sign = [x if x != 0 else 1 for x in [x_sign, y_sign]]

    if distance == None:
        distance = calc_distance(start, end)

    x_diff = __dx(x_sign * distance, slope)
    y_diff = __dy(x_sign * distance, slope, y_sign)

    if is_alt:
        z_slope = calc_slope(start, end, is_alt=is_alt)
        z_diff = __dy(x_sign * distance, z_slope)
    else:
        z_diff = 0.0

    result = (
        origin[0] + x_diff,
        origin[1] + y_diff,
        origin[2] + z_diff
    ) if len(origin) > 2 else (
        origin[0] + x_diff,
        origin[1] + y_diff
    )

    return result

def move_point_test():

    origin = (0,0,0)

    ends = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0)
    ]

    predicts = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0)
    ]

    for index in range(len(ends)):
        result = move_point(origin, None, start=origin, end=ends[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")    

def move_line(line, start=None, end=None, distance=None, is_alt=False):
    return [move_point(x, None, start=start, end=end, distance=distance, is_alt=is_alt) for x in line]

def move_line_test():

    line = [
        (0,0,0),
        (10,0,0)
    ]

    start = (0,0,0)
    ends = [
        (5,0,0),
        (0,5,0),
        (5,5,0)
    ]
    
    predicts = [
        [
            (5,0,0),
            (15,0,0)
        ],
        [
            (0,5,0),
            (10,5,0)
        ],
        [
            (5,5,0),
            (15,5,0)
        ]
    ]

    for index in range(len(ends)):
        result = move_line(line, None, start=start, end=ends[index])
        log_print("Result = {0}".format(result))
        if result != predicts[index]:
            log_print("Diff")
            break
    else:
        log_print("Test Complete [OK]")

def get_ortho_line(origin, slope=None, start=None, end=None, degree=None, dist1=25.0, dist2=25.0):

    x_sign = 1
    y_sign = 1

    # 1. 진행방향 기울기 추출
    if slope == None:
        # - degree 기반 시작/종료 좌표 추출
        if start == None:
            degree = degree % 360
            start = (0,0,0)
            if degree % 90 == 0:
                if degree == 0:
                    end = (1,0,0)
                elif degree == 90:
                    end = (0,1,0)
                elif degree == 180:
                    end = (-1,0,0)
                else:
                    end = (0,-1,0)
            else:
                x = -1.0 if 90 < degree < 270 else 1.0
                end = (x, convert_to_slope(degree) * x, 0)

        slope = calc_slope(start, end)
        x_sign = np.sign(end[0] - start[0])
        y_sign = np.sign(end[1] - start[1])

    # - 기울기가 x 축인 경우
    if slope == 0.0:
        left = (origin[0], origin[1] + dist1 * x_sign, origin[2])
        right = (origin[0], origin[1] - dist2 * x_sign, origin[2])
        return [right, origin, left]
    # - 기울기가 y 축인 경우
    elif slope == float("inf"):
        left = (origin[0] - dist1 * y_sign, origin[1], origin[2])
        right = (origin[0] + dist2 * y_sign, origin[1], origin[2])
        return [right, origin, left]

    ortho_slope = -1.0 / slope

    left = move_point(origin, ortho_slope, distance=dist1 * -y_sign)
    right = move_point(origin, ortho_slope, distance=dist2 * y_sign)

    return [right, origin, left]

def get_ortho_line_test():
    
    import os
    os.system("clear")

    def slope_test():
        origin = (0,0,0)

        slopes = [float("inf"), 1.0, 0.0]
        distances = [1.0, math.sqrt(2) / 2.0, 1.0]
        slope_results = [
            [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
            [(0.5, -0.5, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.5, 0.0)],
            [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        ]

        for index, slope in enumerate(slopes):
            ortho_line = get_ortho_line(origin, slope=slope, dist1=distances[index], dist2=distances[index])
            # log_print("{0} : [{1}]".format(index + 1, slope_results[index] == ortho_line))
            if slope_results[index] != ortho_line:
                log_print("Diff : {0}".format(ortho_line))
                break
        else:
            log_print("Test Complete [OK]")
            return

    def start_end_test():
        origin = (0,0,0)

        points = [
            [(0,0,0), (0,1,0)],
            [(0,0,0), (1,1,0)],
            [(0,0,0), (1,0,1)],
            [(0,0,0), (1,-1,0)],
            [(0,0,0), (0,-1,0)],
            [(0,0,0), (-1,-1,0)],
            [(0,0,0), (-1,0,0)],
            [(0,0,0), (-1,1,0)]
        ]
        distances = [
            1.0, 
            math.sqrt(2) / 2.0, 
            1.0,
            math.sqrt(2) / 2.0, 
            1.0,
            math.sqrt(2) / 2.0, 
            1.0,
            math.sqrt(2) / 2.0 
            ]
        slope_results = [
            [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
            [(0.5, -0.5, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.5, 0.0)],
            [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(-0.5, -0.5, 0.0), (0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],
            [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            [(-0.5, 0.5, 0.0), (0.0, 0.0, 0.0), (0.5, -0.5, 0.0)],
            [(0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)],
            [(0.5, 0.5, 0.0), (0.0, 0.0, 0.0), (-0.5, -0.5, 0.0)],
        ]

        for index, point in enumerate(points):
            ortho_line = get_ortho_line(origin, start=point[0], end=point[1], dist1=distances[index], dist2=distances[index])
            # log_print("{0} : [{1}]".format(index + 1, slope_results[index] == ortho_line))
            if slope_results[index] != ortho_line:
                log_print("Expected : {0}".format(slope_results[index]))
                log_print("Result : {0}".format(ortho_line))
                break
        else:
            log_print("Test Complete [OK]")
            return

    def degree_test():
        
        origin = (0,0,0)
        
        degrees = [
            0,
            90,
            180,
            270,
            360,
            45,
            135,
            225,
            315            
        ]

        distances = [
            1,
            1,
            1,
            1,
            1,
            math.sqrt(2),
            math.sqrt(2),
            math.sqrt(2),
            math.sqrt(2)
        ]

        predicts = [
            [(0,-1,0),(0,0,0),(0,1,0)],
            [(1,0,0),(0,0,0),(-1,0,0)],
            [(0,1,0),(0,0,0),(0,-1,0)],
            [(-1,0,0),(0,0,0),(1,0,0)],
            [(0,-1,0),(0,0,0),(0,1,0)],
            [(1,-1,0),(0,0,0),(-1,1,0)],
            [(1,1,0),(0,0,0),(-1,-1,0)],
            [(-1,1,0),(0,0,0),(1,-1,0)],
            [(-1,-1,0),(0,0,0),(1,1,0)],
        ]

        for index in range(len(degrees)):
            result = get_ortho_line(origin, degree=degrees[index], dist1=distances[index], dist2=distances[index])
            if not all([round(result[x][i], 5) == round(predicts[index][x][i], 5) for x in [0, 2] for i in [0, 1, 2]]):
                log_print("Index : {0}".format(index+1))
                log_print("Predict : {0}".format(predicts[index]))
                log_print("Result : {0}".format(result))
                break
        else:
            log_print("Test Complete [OK]")

    slope_test()
    start_end_test()
    degree_test()

def get_mid(p1, p2):
    """
    2개의 좌표(p1, p2) 중점 반환
    """

    return tuple([(p1[index] + p2[index]) / 2.0 for index in range(len(p1 if len(p1) <= len(p2) else p2))])

def find_parallel(first, second, error_degree=5.0):
    """
    2개의 좌표목록에서 평행한 영역(main)을 반환한다. 
    """

    parallel_parts = []

    # 1. main - sub 분류 (직선 기준)
    (main, sub) = (first, second) if select_straight([first, second]) == first else (second, first)
    part_index = 0

    # 2. sub 의 각 선분마다 main 과의 평행여부 측정
    for sub_line in [sub[index:index+2] for index in range(len(sub) - 1)]:
        # 1) sub 선분에 가장 근접한 main 선분 추출
        main_line = get_closest_segment(get_mid(sub_line[0], sub_line[-1]), main)
        # 2) sub 선분 main 선분 간 각도차이 측정
        degree_diff = calc_curve_diff(sub_line, compare_points=main_line)
        # - 각도차가 오차(error_degree) 미만인 경우 : 평행으로 판정
        if degree_diff < error_degree:
            if len(parallel_parts) - 1 < part_index:
                parallel_parts.append([])
                parallel_parts[part_index].append(sub_line[0])
            parallel_parts[part_index].append(sub_line[-1])
        # - 평행 판정이 아닌 경우
        else:
            if len(parallel_parts) - 1 >= part_index:
                part_index += 1

    return parallel_parts

def find_parallel2(first, second, error_degree=5.0):
    
    seg_list = []

    (main, sub) = (first, second) if select_straight([first, second], is_index=True) == 0 else (second, first)
    
    for sub_seg in [sub[index:index+2] for index in range(len(sub) - 1)]:
        sub_mid = get_mid(sub_seg[0], sub_seg[-1])
        main_seg = get_closest_segment(sub_mid, main)

        sub_deg = calc_degree(sub_seg[0], sub_seg[-1])
        main_deg = calc_degree(main_seg[0], main_seg[-1])

        deg_diff = abs(main_deg - sub_deg)

        if deg_diff < error_degree:
            seg_list.append(sub_seg)
    else:
        pass

    # todo : 평행영역을 main 기준으로 추출
    # todo : 평행영역 조건
    # todo : 1) 2개의 선분 간 각도차 이하
    # todo : 2) 2개의 선분 간 수직 포함영역 일정 비율 이상

    return 

def find_parallel_test():
    
    main = [(x, 0, 0) for x in range(0, 11)]
    
    sub_list = [
        [(x, 4, 0) for x in range(0, 11)],
        [(x, 1, 0) for x in range(0, 11)],
        [(x, 7, 0) for x in range(0, 11)],
        [
            (0,4,0),
            (4,4,0),
            (6,8,0),
            (10,8,0)
        ],
        [
            (0,4,0),
            (4,4,0),
            (4,8,0),
            (6,8,0),
            (6,4,0),
            (10,4,0)
        ]
    ]

    predicts = [
        False,
        True,
        True,
        True,
        False
    ]

    for index in range(len(sub_list)):
        sub = sub_list[index]
        result = find_parallel(main, sub)
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Result = {0}".format(result))
            log_print("Predict = {0}".format(predicts[index]))
            break
    else:
        log_print("Test Complete [OK]")

def calc_parallel(first, second, error_degree=5.0):
    
    # 1. 평행 영역 추출
    parallel_parts = find_parallel(first, second, error_degree=error_degree)

    # 2. 평행 영역 길이 추출
    length = reduce(lambda x, y : x + y, [calc_length(part) for part in parallel_parts], 0.0)

    return length 

def get_point_on_points(points, target_length, division=None):
    """
    좌표목록(points) 길이 기준 특정 지점의 좌표 반환
    """

    # - 좌표목록 입력 에러
    if len(points) < 1:
        return None

    # - 좌표 개수가 1개 이하인 경우
    if len(points) < 2:
        return points[0]

    # 1. 좌표목록 길이 총합 추출
    total_length = calc_length(points)

    # - 총 길이를 나눈 값이 목표 지점이라면
    if division != None:
        target_length = float(total_length) / float(division)

    # - 목표 거리가 길이 총합보다 큰 경우
    if target_length > total_length:
        log_print("Target length is bigger than total length")
        return None

    # 2. 첫번째 좌표를 시작으로 목표 좌표 계산
    length_sum = 0.0
    for index in range(len(points) - 1):
        # 1) 현 순번의 선분 길이 추출
        curr_len = calc_distance(points[index], points[index+1])
        # - 지금까지의 길이 합 + 현 선분 길이 > 목표 길이 인 경우
        if length_sum + curr_len >= target_length:
            # - 현 선분 상에 위치한 목표 좌표를 추출
            target_point = move_point(points[index], None, start=points[index], end=points[index+1], distance=(target_length - length_sum))
            # - 좌표 반환
            return target_point
        # - 지금까지의 길이 합 + 현 선분 길이 < 목표 길이 인 경우
        else:
            # - 현 선분의 길이를 길이 합에 더한다. (다음 선분으로 진행)
            length_sum += curr_len

    # - 좌표 추출 실패
    log_print("Cannot find target point")
    return None

def get_point_on_points(points, length=None, division=None):
    
    point = None

    total = calc_length(points)

    if division != None:
        length = total / float(division)

    if length > total:
        warning_print("Target length is longer than total ({0} / {1})".format(length, total))
        return point

    _length = 0.0
    for seg in [points[index:index+2] for index in range(len(points)-1)]:
        dist = calc_distance(seg[0], seg[-1])
        if _length + dist >= length:
            point = move_point(seg[0], None, start=seg[0], end=seg[-1], distance=length-_length)
            break
        _length += dist

    return point

def get_point_on_points_test():

    points = [
        (0,0,0),
        (1,0,0),
        (2,0,0),
        (3,0,0),
        (4,0,0),
        (5,0,0),
        (6,0,0),
        (7,0,0),
        (8,0,0),
        (9,0,0),
        (10,0,0),
    ]

    divisions = [1, 2, 3, 4, 5]

    predicts = [
        (10,0,0),
        (10.0/2.0,0,0),
        (10.0/3.0,0,0),
        (10.0/4.0,0,0),
        (10.0/5.0,0,0),
    ]

    for index, division in enumerate(divisions):
        result = get_point_on_points(points, division=division)
        if not all([round(result[i], 5) == round(predicts[index][i], 5) for i in range(3)]):
            log_print("Index : {0}".format(index))
            log_print("Result : {0}".format(result))
            log_print("Predict : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 1 Complete [OK]")

    points = [
        (0,0,0),
        (5,5,0),
        (10,0,0),
    ]

    divisions = [1, 2, 3, 4, 5]

    predicts = [
        (10,0,0),
        (10.0/2.0,(5.0 * 2 / 2.0),0),
        (10.0/3.0,(5.0 * 2 / 3.0),0),
        (10.0/4.0,(5.0 * 2 / 4.0),0),
        (10.0/5.0,(5.0 * 2 / 5.0),0),
    ]

    for index, division in enumerate(divisions):
        result = get_point_on_points(points, division=division)
        if not all([round(result[i], 5) == round(predicts[index][i], 5) for i in range(3)]):
            log_print("Index : {0}".format(index))
            log_print("Result : {0}".format(result))
            log_print("Predict : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 2 Complete [OK]")

def sample_points(line, count=None):
    """
    선에서 일정 간격마다 추출한 좌표목록을 반환
    - 시작/종료 좌표는 유지
    """

    samples = []

    total_length = calc_length(line)

    # 샘플링 개수 (기본 1m)
    if count == None:
        count = int(total_length) - 1

    # - 샘플링 간격
    interval = total_length / float(count)

    dist_sum = 0.0
    queue = list(line)

    while len(queue) >= 2:

        if len(samples) >= count - 1:
            break

        seg, queue = queue[:2], queue[2:]
        
        dist = calc_distance(seg[0], seg[-1])

        if interval - (dist_sum + dist) < 0.00001:
            sample = move_point(seg[0], None, start=seg[0], end=seg[-1], distance=interval-dist_sum)
            samples.append(sample)
            dist_sum = 0.0
            queue = [sample, seg[-1]] + queue
        else:
            dist_sum += dist
            queue = [seg[-1]] + queue            

    # - 시작/종료 좌표 유지
    samples = line[:1] + samples  + line[-1:]

    return samples

def sample_points_test():

    line = [(0,0,0), (10,0,0)]
    
    counts = [10, 5, 2, 1]

    predicts = [
        [
            (0,0,0),
            (1,0,0),
            (2,0,0),
            (3,0,0),
            (4,0,0),
            (5,0,0),
            (6,0,0),
            (7,0,0),
            (8,0,0),
            (9,0,0),
            (10,0,0),
        ],
        [
            (0,0,0),
            (2,0,0),
            (4,0,0),
            (6,0,0),
            (8,0,0),
            (10,0,0),            
        ],
        [
            (0,0,0),
            (5,0,0),
            (10,0,0),
        ],
        [
            (0,0,0),
            (10,0,0),
        ],
    ]
    
    for index, count in enumerate(counts):
        result = sample_points(line, count=count)
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predict : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 1 Complete [OK]")
    
    line = [(0,0,0), (5,5,0), (10,5,0)]
    
    counts = [10, 5, 2, 1]

    predicts = [
        [
            (0, 0, 0), 
            (0.8535533905932736, 0.8535533905932736, 0.0), 
            (1.7071067811865472, 1.7071067811865472, 0.0), 
            (2.560660171779821, 2.560660171779821, 0.0), 
            (3.4142135623730945, 3.4142135623730945, 0.0), 
            (4.267766952966368, 4.267766952966368, 0.0), 
            (5.171572875253808, 5.0, 0.0), 
            (6.378679656440356, 5.0, 0.0), 
            (7.585786437626904, 5.0, 0.0), 
            (8.792893218813452, 5.0, 0.0), 
            (10.0, 5.0, 0.0)
        ], 
        [
            (0, 0, 0), 
            (1.7071067811865472, 1.7071067811865472, 0.0), 
            (3.4142135623730945, 3.4142135623730945, 0.0), 
            (5.171572875253809, 5.0, 0.0), 
            (7.585786437626904, 5.0, 0.0), 
            (10.0, 5.0, 0.0)            
        ],
        [
            (0, 0, 0), 
            (4.267766952966369, 4.267766952966369, 0.0), 
            (10.0, 5.0, 0.0)
        ],
        [
            (0,0,0),
            (10,5,0),
        ],
    ]
    
    for index, count in enumerate(counts):
        result = sample_points(line, count=count)
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Result : {0}".format(result))
            log_print("Predict : {0}".format(predicts[index]))
            break
    else:
        log_print("Test 2 Complete [OK]")

def check_is_left(a, b, c):
    """
    특정 좌표(c)가 선(a - b)의 좌 / 우 어디에 위치하는지 구분
    - 검사 대상이 선 위에 존재하는 경우 False (우측으로 판단)
    """

    if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0:
        return True
    return False

def sort_line_by_line(line, standard, is_index=False):
    """
    1개의 좌표목록(line)을 기준선(standard)에 기반해 정렬
    - 기준선의 시작 좌표에 가까운 순으로 정렬한다.
    """

    def __get_index(x, standard):
        return standard.index(get_closest_point(x, standard))

    def __get_distance(x, standard):
        # 1. 최근접 좌표 추출
        origin = get_closest_point(x, standard)
        # 2. 기준선 선분 추출
        segment = get_closest_segment(origin, standard, is_index=True)
        # 3. 기준선 좌표 - 특정 좌표까지의 거리(부호 O) 추출
        return calc_distance_from_point(origin, x, start=segment[0], end=segment[-1], is_sign=True)

    sorted_points = sorted(line, key=lambda x : (
        # 1. 각 선의 종료점에 가장 가까운 기준선 좌표 목차에 따라 정렬
        __get_index(x, standard),
        # 2. 1번 결과가 동일한 경우 : 가장 가까운 기준선 좌표와의 거리에 따라 정렬
        __get_distance(x, standard)
        ))

    return sorted_points if not is_index else [line.index(x) for x in sorted_points]

def sort_line_by_line_test():

    standards = [
        [
            (1, 2, 0),
            (2, 1, 0),
            (3, 0, 0),
            (2, -1, 0),
            (1, -2, 0),
            (0, -3, 0),
            (-1, -2, 0),
            (-2, -1, 0),
            (-3, 0, 0),
            (-2, 1, 0),
            (-1, 2, 0)
        ],
        [
            (0,0,0),
            (10,0,0)
        ]
    ]

    lines = [
        [move_point(x, None, start=(0,0,0), end=x, distance=calc_distance((0,0,0), x) * 0.2) for x in standards[0]],
        [
            (0,1,0), (1,4,0), (7,2,0), (9,0,0),
            (2,-1,0), (3,-7,0), (6,-3,0), (8,-4,0)
        ]
    ]

    predicts = [
        list(lines[0]),
        sorted(lines[1], key=lambda x : x[0])
    ]

    for index in range(len(lines)):
        result = sort_line_by_line(lines[index], standards[index])
        if result != predicts[index]:
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def sort_lines_by_line(lines, standard, is_index=False):

    sorted_lines = []

    # 1. 선 목록을 선 시작좌표(line[0]) 기준으로 정렬한다.
    _lines = list(lines)
    # 1) 정렬된 시작좌표를 순서대로 추출
    for point in sort_line_by_line([x[0] for x in lines], standard):
        # 2) 추출된 시작좌표를 첫번째 좌표로 가지는 선을 검색
        # - 시작좌표가 동일한 경우 목록에서 첫번째로 발견된 선이 검색된다.
        line = next((x for x in _lines if x[0] == point), None)
        # 3) 정렬 목록에 추가
        sorted_lines.append(line)
        # 4) 한번 검색된 선 제거
        # - 시작좌표가 동일한 경우 중복된 선 검색을 방지하기 위함
        _lines.pop(_lines.index(line))

    for line in sorted_lines:
        log_print("Test = {0}".format(line))

    return sorted_lines if not is_index else [lines.index(x) for x in sorted_lines]

def sort_lines_by_line_test():

    lines_list = [
        [
            [(0,1,0), (5,1,0)],
            [(1,2,0), (3,2,0)],
            [(2,3,0), (10,3,0)]
        ],
        [
            [(0,1,0), (5,1,0)],
            [(0,1,0), (4,1,0)],
            [(0,1,0), (7,1,0)]
        ],
        [
            [(0,2,0), (6,2,0)],
            [(0,1,0), (5,1,0)], 
            [(0,3,0), (3,3,0)]
        ]
    ]

    standards = [
        [(0,0,0), (10,0,0)],
        [(x,0,0) for x in range(0, 11)],
        [(0,0,0), (10,0,0)]
    ]

    predicts = [
        [
            lines_list[0][0],
            lines_list[0][1],
            lines_list[0][2]
        ], 
        [
            lines_list[1][0],
            lines_list[1][1],
            lines_list[1][2]
        ],
        [
            lines_list[2][0],
            lines_list[2][1],
            lines_list[2][2]
        ]
    ]

    for index in range(len(lines_list)):
        result = sort_lines_by_line(lines_list[index], standards[index])
        if result != predicts[index]:
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")
    
def parse_points(points, parse_lines, is_debug=False):
    
    def check_point_in_segment(p, seg):
        """
        동일한 좌표를 기반한 수직선 생성 시 소수점 오차 발생 가능
        """

        for index, _p in enumerate(seg):
            if calc_distance(p, _p) < 0.001:
                if is_debug:
                    log_print("Point on segment : {0}".format(index))
                return True
        return False

    data_list = []

    _points = list(points)

    while len(_points) > 0:

        for index in range(len(_points) - 1):
            # 1. 선분 추출
            seg = _points[index:index+2]

            if is_debug:
                log_print("Point : {0}".format(index))

            # 2. 선분-분할선 교차점 목록 추출
            intersect_list = []
            for idx, line in enumerate(parse_lines):
                # - 중복 분할 방지 
                if idx not in [data["index"] for data in data_list]:
                    intersect_p = get_intersection_on_points(line, seg)
                    # - 교차점 추가
                    if intersect_p != None:
                        intersect_list.append({
                            "index" : idx,
                            "point" : intersect_p,
                        })
                        if is_debug:
                            log_print("Intersect : {0} - {1}".format(idx, intersect_p))

            # - 선분 시작에 해당하는 교차점 제외
            # - 선분과 분할선의 교차점은 시작을 제외하며 종료를 포함 (= 좌표목록 시작점과 교차하는 경우 제외)            
            intersect_list = [data for data in intersect_list if data["point"] != seg[0]]

            # 3. 교차점 목록에서 선분 시작점에 최근접 교차점 추출
            if len(intersect_list) > 0:
                # - 1) 최근접 교차점 추출
                intersect_data = sorted(intersect_list, key=lambda x : calc_distance(seg[0], x["point"]))[0]

                # - 교차점이 선분 시작/종료 좌표에 해당하지 않는 경우
                if not check_point_in_segment(intersect_data["point"], seg):
                    # - 좌표목록에 교차점 삽입
                    _points.insert(index+1, intersect_data["point"])
                    if is_debug:
                        log_print("Add intersect")
                
                    part, _points = _points[:index+2], _points[index+1:] 
                else:
                    if calc_distance(seg[0], intersect_data["point"]) < 0.001:
                        part, _points = _points[:index+1], _points[index:] 
                    else:
                        part, _points = _points[:index+2], _points[index+1:] 
                    

                data_list.append({
                    "index" : intersect_data["index"], 
                    "points" : part,
                    "is_prev" : True,
                })
                if is_debug:
                    log_print("Points : {0}".format(data_list[-1]["index"]))
                break
        # - 모든 선분 검사 후 좌표목록이 남은 경우
        else:
            # - 남은 좌표가 2개 이상인 경우 (= 분할선이 종료좌표와 동일한 경우 제외)
            if len(_points) > 1:
                # - 추출된 분할정보가 1개 이상인 경우 (= 분할선 모두 좌표목록과 교차하지 않는 경우 제외)
                if len(data_list) > 0:
                    data_list.append({
                        # "index" : max([data["index"] for data in data_list]) + 1,
                        "index" : data_list[-1]["index"],
                        "points" : _points,
                        "is_prev" : False,
                    })
                    if is_debug:
                        log_print("Points (Extra) : {0}".format(data_list[-1]["index"]))
            break

    # 분할정보 목록은 좌표목록의 순번에 비례한다. (분할 순번과 관계 X)

    return data_list

def parse_points_test():

    os.system("clear")

    points = [
        (1, 5, 0), 
        (2, 4, 0), 
        (3, 3, 0), 
        (4, 2, 0), 
        (5, 1, 0), 
        (5, -1, 0), 
        (4, -2, 0), 
        (3, -3, 0), 
        (2, -4, 0), 
        (1, -5, 0), 
        (-1, -5, 0), 
        (-2, -4, 0), 
        (-3, -3, 0), 
        (-4, -2, 0), 
        (-5, -1, 0),
        (-5, 1, 0),
        (-4, 2, 0),
        (-3, 3, 0), 
        (-2, 4, 0), 
        (-1, 5, 0)
    ]

    line_packs = [
        [
            [(0, 0, 0), (0, -5, 0), (0, -10, 0)], 
            [(0, 0, 0), (5,  0, 0), (10,  0, 0)], 
            [(0, 0, 0), (-5, 0, 0), (-10, 0, 0)] 
        ],
        [
            [(0,0, 0), (6,  6, 0)], 
            [(0,0, 0), (6, -6, 0)], 
            [(0,0, 0), (-6,-6, 0)]
        ],
        [
            [(0,-0.1, 0), (10, -0.1, 0)], 
            [(0, 0.1, 0), (10,  0.1, 0)]
        ]
    ]
    
    predicts = [
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0), (4, 2, 0), (5, 1, 0), (5.0, -0.0, 0)],
            [(5.0, -0.0, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0), (2, -4, 0), (1, -5, 0), (-0.0, -5.0, 0.0)],
            [(-0.0, -5.0, 0.0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5.0, 0.0, 0)],
            [(-5.0, 0.0, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)]
        ],
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0)],
            [(3, 3, 0), (4, 2, 0), (5, 1, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0)],
            [(3, -3, 0), (2, -4, 0), (1, -5, 0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0)],
            [(-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)]
        ],
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0), (4, 2, 0), (5, 1, 0), (5.0, 0.1, 0)],
            [(5.0, 0.1, 0), (5.0, -0.1, 0)],
            [(5.0, -0.1, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0), (2, -4, 0), (1, -5, 0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)]
        ]
    ]

    for index in range(len(line_packs)):
        result = parse_points(points, line_packs[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict :")
            for points in predicts[index]:
                log_print(" - {0}".format(points))
            for points in result:
                log_print(" - {0}".format(points))
            break
    else:
        log_print("Test Complete [OK]")

def parse_points_test2():

    os.system("clear")

    points = [
        (1, 5, 0), 
        (2, 4, 0), 
        (3, 3, 0), 
        (4, 2, 0), 
        (5, 1, 0), 
        (5, -1, 0), 
        (4, -2, 0), 
        (3, -3, 0), 
        (2, -4, 0), 
        (1, -5, 0), 
        (-1, -5, 0), 
        (-2, -4, 0), 
        (-3, -3, 0), 
        (-4, -2, 0), 
        (-5, -1, 0),
        (-5, 1, 0),
        (-4, 2, 0),
        (-3, 3, 0), 
        (-2, 4, 0), 
        (-1, 5, 0)
    ]

    line_packs = [
        [
            [(0, 0, 0), (0, -5, 0), (0, -10, 0)], 
            [(0, 0, 0), (5,  0, 0), (10,  0, 0)], 
            [(0, 0, 0), (-5, 0, 0), (-10, 0, 0)] 
        ],
        [
            [(0,0, 0), (6,  6, 0)], 
            [(0,0, 0), (6, -6, 0)], 
            [(0,0, 0), (-6,-6, 0)]
        ],
        [
            [(0,-0.1, 0), (10, -0.1, 0)], 
            [(0, 0.1, 0), (10,  0.1, 0)]
        ]
    ]
    
    predicts = [
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0), (4, 2, 0), (5, 1, 0), (5.0, -0.0, 0)],
            [(5.0, -0.0, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0), (2, -4, 0), (1, -5, 0), (-0.0, -5.0, 0.0)],
            [(-0.0, -5.0, 0.0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5.0, 0.0, 0)],
            [(-5.0, 0.0, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)],
        ],
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0)],
            [(3, 3, 0), (4, 2, 0), (5, 1, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0)],
            [(3, -3, 0), (2, -4, 0), (1, -5, 0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0)],
            [(-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)]
        ],
        [
            [(1, 5, 0), (2, 4, 0), (3, 3, 0), (4, 2, 0), (5, 1, 0), (5.0, 0.1, 0)],
            [(5.0, 0.1, 0), (5.0, -0.1, 0)],
            [(5.0, -0.1, 0), (5, -1, 0), (4, -2, 0), (3, -3, 0), (2, -4, 0), (1, -5, 0), (-1, -5, 0), (-2, -4, 0), (-3, -3, 0), (-4, -2, 0), (-5, -1, 0), (-5, 1, 0), (-4, 2, 0), (-3, 3, 0), (-2, 4, 0), (-1, 5, 0)]
        ]
    ]

    for index in range(len(line_packs)):
        data_list = parse_points(points, line_packs[index])
        result = [data["points"] for data in data_list]
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict :")
            for points in predicts[index]:
                log_print(" - {0}".format(points))
            log_print("Result : ")
            for points in result:
                log_print(" - {0}".format(points))
            break
    else:
        log_print("Test Complete [OK]")

def check_between(target, points):
    """
    특정 좌표(target)가 좌표목록(points)의 시작 ~ 종료 수직 범위 내에 존재하는지 검사
    """

    start_line = get_ortho_line(points[0], start=points[0], end=points[-1])
    end_line = get_ortho_line(points[-1], start=points[-2], end=points[-1])

    check1 = not check_is_left(start_line[0], start_line[-1], target) 
    check2 = check_is_left(end_line[0], end_line[-1], target)

    return check1 and check2

def check_between_test():

    points = [
        (0,0,0),
        (10,0,0)
    ]

    targets = [
        (0,0,0),
        (10,0,0),
        (5,0,0),
        (7,5,0),
        (2,-2,0),
        (-1,0,0),
        (11,0,0),
    ]

    predicts = [
        True,
        False,
        True,
        True,
        True,
        False,
        False,
    ]

    for index in range(len(targets)):
        result = check_between(targets[index], points)
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def improve_points_density(points, density=1.0):
    """
    좌표목록을 구성하는 선분 사이에 추가 좌표 삽입
    """

    improved_points = []
    # 1. 각 선분 별 밀도 증가
    for segment in [points[index:index+2] for index in range(len(points) - 1)]:
        # 1) 선분 길이 측정
        length = calc_distance(segment[0], segment[-1])
        # 2) 선분 길이에 비례한 추가 좌표 개수 계산
        add_count = int(length / density) 
        # 3) 좌표 추가
        improved_points += (
            [segment[0]] + 
            [move_point(segment[0], None, start=segment[0], end=segment[-1], distance=density * i) for i in range(1, add_count)]
        )
        # - 마지막 선분인 경우 선분 끝 좌표 추가
        if segment[-1] == points[-1]:
            improved_points.append(segment[-1])

    return improved_points

def improve_points_density_test():

    points = [
        (0,0,0),
        (10,0,0),
        (15,0,0)
    ]

    density = math.sqrt(1)

    predict = [(x, 0, 0) for x in range(16)]

    result = improve_points_density(points, density=density)
    if result != predict:
        log_print("Predict : {0}".format(predict))
        log_print("Result : {0}".format(result))
    else:
        log_print("Test Complete [OK]")

def get_inner_segment(point, points, is_debug=False):
    """
    좌표목록(points)을 구성하는 선분 중 특정 좌표(point)를 포함하는 선분을 반환 
    """

    if is_debug:
        log_print("Test 1 = {0}".format(point))
        log_print("Test 2 = {0}".format(points))

    for segment in [points[index:index+2] for index in range(len(points) - 1)]:
        if check_point_on_line(segment, point, is_debug=is_debug):
            return segment
            
    return None

def get_symmetry_point(origin, line):

    # 1. 기울기 / 절편 추출
    slope = calc_slope(line[0], line[-1])

    if slope == 0:
        return (
            origin[0],
            origin[1] + 2 * (line[0][1] - origin[1]),
            origin[2]
        )
    elif slope == float("inf"):
        return (
            origin[0] + 2 * (line[0][0] - origin[0]),
            origin[1],
            origin[2]
        )

    bias = line[0][1] - slope * line[0][0]
    log_print("Test = {0}".format((slope, bias)))

    # 2. 계산
    return (
        ((1 - slope**2)*origin[0] + 2*slope*origin[1] - 2*slope*bias) / (1+slope**2),
        (2*slope*origin[0] - (1-slope**2)*origin[1] + 2*bias) / (1+slope**2),
        origin[2]
    )

def get_symmetry_point_test():

    origin = (0,0,0)

    lines = [
        [(1, -1, 0), (1, 1, 0)],
        [(-1,1,0), (1,1,0)],
        [(-1,-1,0), (-1,1,0)],
        [(-1,-1,0), (1,-1,0)],
        [(2,0,0,), (0,2,0)],
        [(4,0,0), (0,2,0)],
        [(2,0,0), (0,3,0)],
    ]

    predicts = [
        (2,0,0),
        (0,2,0),
        (-2,0,0),
        (0,-2,0),
        (2,2,0),
        (4,2,0),
        (2,3,0),
    ]

    for index in range(len(lines)):
        result = get_symmetry_point(origin, lines[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def simplify_polygon(points, elapse_dist=0.2):
    """
    더글라스 패커(Douglas Peuker) 알고리즘을 사용한 도형 단순화
    1. 시작-종료 선 추출 (초기값) : 선분 목록 / 유지좌표에 추가
    2. 선분에서 수직거리가 가장 먼 좌표 추출
    3. 단순화 검사
    - 임계값보다 큰 경우 : 유지 좌표에 추가 / 신규 선 추출
    - 임계값보다 작은 경우 : 제거
    - 검사 쌍 기록 (이미 검사한 조합은 신규 선 생성시 생략)
    > 2 ~ 3 반복 (더이상 신규 유지 좌표가 없을 때까지)
    """    

    def get_new_line(new_point, origin_points, simplified_points):
        """
        유지 좌표목록에 추가된 좌표에서 신규 선분을 추출한다.
        - 선분 기록에 존재하는 조합은 제외한다.
        - 신규 좌표(new_point)에서 전/후로 가장 가까운 2개의 유지 좌표에 대해서만 신규 선분을 생성한다.
        """

        new_index = origin_points.index(new_point)
        
        prev_index = -1 
        next_index = float("inf")
        for point in simplified_points:
            point_index = origin_points.index(point)
            if point_index < new_index and point_index > prev_index:
                prev_index = point_index
            elif point_index > new_index and point_index < next_index:
                next_index = point_index

        prev_point = origin_points[prev_index]        
        next_point = origin_points[next_index]        

        prev_line = [prev_point, new_point]
        next_line = [new_point, next_point]

        return prev_line, next_line

    if len(points) < 3:
        return points

    if points[0] == points[-1]:
        origin_points = points[:-1]
    else:
        origin_points = list(points)

    # 유지 좌표목록
    simplified_points = []
    # 선분 큐
    line_queue = []

    # 1. 초기 선분 추출
    init_line = origin_points[:1] + origin_points[-1:]
    # 선분 큐에 추가
    line_queue.append(init_line)
    # 유지 좌표목록에 추가
    simplified_points += init_line

    # 더이상 추출할 선분 조합이 없는경우 종료
    while len(line_queue) > 0:
        # 2-1. 선분 추출
        curr_line = line_queue.pop(0)

        # 2-2. 선분에 속한 좌표 및 유지 좌표목록에 속한 좌표를 제외한 좌표 추출
        # - 현재 선분 사이의 목차에 속한 좌표만 추출
        start_index, end_index = sorted([origin_points.index(x) for x in curr_line])
        target_points = [x for x in origin_points[start_index+1:end_index] if x not in curr_line and x not in simplified_points and origin_points.index(x)]
        # - 해당하는 좌표가 없을 시 종료
        if len(target_points) < 1:
            continue

        # 2-3. 대상 좌표들을 먼 거리순으로 정렬
        sorted_points = sorted(target_points, key=lambda x : calc_distance_from_line(curr_line, x), reverse=True)
        # 2-4. 거리 측정
        dist = calc_distance_from_line(curr_line, sorted_points[0]) 
        # 2-5. 유지 검사
        # - 임계값보다 거리가 크다면 유지
        if dist > elapse_dist:
            # 유지 좌표목록에 추가
            simplified_points.append(sorted_points[0])
            # 신규 선분 추가
            prev_line, next_line = get_new_line(sorted_points[0], origin_points, simplified_points)
            line_queue += [prev_line, next_line]
        # - 임계값보다 거리가 작다면 제거
        else:
            del origin_points[origin_points.index(sorted_points[0])]

    result_points = sorted(simplified_points, key=lambda x : origin_points.index(x))

    return result_points

def simplify_polygon2(points, elapse_dist=0.2):
    """
    (N, N+2) 선분과 N+1 좌표 간 수직 거리가 임계값(elapse_dist)을 초과하는 경우 N+1 좌표 제외
    """

    if len(points) < 3:
        return points

    _points = []

    index = 0
    indices = []

    while index < len(points) - 2:
        A = points[index]
        B = points[index+1]
        C = points[index+2]
        if calc_distance_from_line([A, C], B) > elapse_dist:
            indices.append(index+1)
            index += 2
        else:
            index += 1

    for point in points:
        if points.index(point) not in indices:
            _points.append(point)

    return _points    

def check_inside(origin, polygon):

    if polygon[-1] != polygon[0]:
        polygon = polygon + [polygon[0]]

    half_line = [origin, (max([x[0] for x in polygon]) + 1.0, origin[1], origin[2])]

    count = 0
    for seg in [polygon[index:index+2] for index in range(len(polygon)-1)]:
        if get_intersection_on_points(half_line, seg) != None:
            count += 1

    if count % 2 != 0:
        return True
    return False

def check_inside_test():

    origin = (0,0,0)

    polygons = [
        [
            (1,1,0),
            (-1,1,0),
            (-1,-1,0),
            (1,-1,0)
        ],
        [
            (1,0,0), 
            (1,1,0),
            (0,1,0)
        ],
        [
            (1,.5,0),
            (1,1,0),
            (-1,1,0),
            (-1,-1,0),
            (1,-1,0),
            (1,-.5,0),
            (-1,-.5,0),
            (-1,.5, 0)
        ],
    ]

    predicts = [
        True,
        False,
        False,    
    ]

    for index in range(len(polygons)):
        result = check_inside(origin, polygons[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test complete [OK]")

def get_convex(points):
    """
    x, y 평면 상에서 Convex Hull 반환
    - 반환되는 좌표는 시작과 끝이 동일하지 않음 (연결 X)
    """

    # - 삼각형 이하(선)는 그대로 반환
    if len(points) < 3:
        return points

    # 1. 3차원 좌표인 경우 2차원(x, y) 좌표로 변환
    # - ConvexHull API 필요조건
    _points = [x[:2] for x in points] if len(points) > 2 else points

    # 2. 변환 
    convex = [_points[x] for x in list(ConvexHull(_points).vertices)]

    # 3. z 조정
    if len(points[0]) > len(convex[0]):
        convex = [(x[0], x[1], .0) for x in convex]

    return convex

def get_convex_test():
    
    polygons = [
        [
            (0,0,0),
            (.5,0,0),
            (1,0,0),
            (1,1,0),
            (0,1,0)
        ],
        [
            (0,0,0),
            (2,0,0),
            (1,1,0)
        ],
        [
            (0,0,0),
            (2,0,0),
            (1,1,0),
            (2,2,0),
            (0,2,0)            
        ]
    ]

    predicts = [
        [
            (0,0),
            (1,0),
            (1,1),
            (0,1)
        ],
        [
            (0,0),
            (2,0),
            (1,1)
        ],
        [
            (0,0),
            (2,0),
            (2,2),
            (0,2)
        ]
    ]

    for index in range(len(polygons)):
        result = get_convex([x[:2] for x in polygons[index]])
        if not all([x in predicts[index] for x in result]):
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def parse_parallel(polygon):
    """
    시작과 끝이 동일한 도형(polygon)을 평행한 방향으로 분할
    """

    def __rotate(point, radian):

        # - 회전각도가 0 인 경우 생략
        if radian == 0:
            return point

        # 1. 좌표 변환
        _point = np.array(point)
        # 2. 회전행렬 추출
        rotation = np.array([
            [math.cos(radian), -math.sin(radian)],
            [math.sin(radian), math.cos(radian)]
        ])
        # 3. 회전
        _point = _point.dot(rotation)
        # 4. 반환
        return tuple([round(x, 5) for x in _point])

    def __parse(parse_line, points):
        """
        좌표목록(points)을 2개의 집합으로 분할
        """

        # 1. 분할
        left, right = [[x for x in points if check_is_left(parse_line[0], parse_line[-1], x) == check] for check in [True, False]]

        # 2. 원본 순서대로 정렬
        left, right = [sorted(bound, key=lambda x : points.index(x)) for bound in [left, right]]

        # left_points 가 비연속인 경우
        if set([points[0], points[-1]]).issubset(left):
            start_index = left.index(points[points.index(right[-1]) + 1])
            rear = left[start_index:]
            front = left[:start_index]
            left = rear + front
        # right 가 비연속인 경우
        elif set([points[0], points[-1]]).issubset(right):
            start_index = right.index(points[points.index(left[-1]) + 1])
            rear = right[start_index:]
            front = right[:start_index]
            right = rear + front

        return left, right

    # 1. 도형의 Convex Hull 추출
    # - 반환되는 좌표목록의 시작/종료 좌표는 동일하지 않다. (연결 X)
    convex = get_convex(polygon)

    # - Convex Hull 단순화
    convex = simplify_polygon(convex, elapse_dist=.2)

    # 2. 도형의 선분 추출
    segments = []
    for index in range(len(convex)):
        if index < len(convex) - 1:
            seg = convex[index:index+2]
        else:
            seg = [convex[-1], convex[0]]
        segments.append(seg)
    # segments = [[convex[index], convex[index+1 if index <= len(convex)-2 else 0]] for index in range(len(convex))]

    # 3. 도형의 각 선분에서 Radian 추출
    angles = []
    for seg in segments:
        deg = calc_degree(seg[0], seg[-1])
        angle = math.radians(deg)
        angles.append(angle)
    # angles = [math.radians(calc_degree(x[0], x[1])) for x in segments]

    # 4. 도형의 각 선분 별 최소사각형 영역 비교
    min_area = float("inf")
    min_index = 0
    min_width = 0
    min_seg = segments[0]
    for index in range(len(segments)):
        seg = segments[index]
        # 1) Convex Hull 회전
        rotated = [__rotate((x[0], x[1]), angles[index]) for x in convex]
        # 2) 최소/최대 x, y 추출
        min_x, min_y, max_x, max_y = [getter([x[i] for x in rotated]) for getter in [min, max] for i in [0, 1]]
        # 3) 최소사각형 면적 계산
        area = (max_x - min_x) * (max_y - min_y)
        # - 면적이 기록된 최소 면적 이하인 경우
        if area <= min_area:
            # - 동일한 경우
            if area == min_area:
                # - 이전 선분보다 긴 경우 갱신
                if calc_distance(min_seg[0], min_seg[-1]) < calc_distance(seg[0], seg[-1]):
                    min_area = area
                    min_index = index
                    min_width = max_x - min_x
                    min_seg = seg
            # - 면적이 미만인 경우
            else:
                min_area = area
                min_index = index
                min_width = max_x - min_x
                min_seg = seg

    # 5. 기준선 추출
    parse_line = segments[min_index]

    # 6. 기준선 길이 조정
    multiple = (min_width / calc_distance(parse_line[0], parse_line[-1])) * 2
    parse_line = scale(parse_line, multiple * 2)

    # 7. 기준선 위치 조정
    start = get_mid(parse_line[0], parse_line[-1])
    end = get_center(convex)
    parse_line = move_line(parse_line, start=start, end=end)

    # 8. 분할
    left, right = __parse(parse_line, convex)

    return left, right

def scale(polygon, multiple):
    """
    도형의 크기 증감
    """

    # 1. 중심점 추출
    center_point = get_center(polygon)

    # 2. 확장 / 축소
    _polygon = [move_point(center_point, None, start=center_point, end=x, distance=calc_distance(center_point, x) * multiple) for x in polygon]

    return _polygon

def scale_test():

    polygon = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0)
    ]

    polygon = [
        (1,0,0),
        (0,0,0),
        (-1,0,0),
    ]

    multiples = [
        2, .5, 1
    ]

    predicts = [
        [
            (2,0,0),
            (0,2,0),
            (-2,0,0),
            (0,-2,0)
        ],
        [
            (.5,0,0),
            (0,.5,0),
            (-.5,0,0),
            (0,-.5,0)
        ],
        [
            (1,0,0),
            (0,1,0),
            (-1,0,0),
            (0,-1,0)
        ],
    ]

    predicts = [
        [
            (2,0,0),
            (0,0,0),
            (-2,0,0)
        ],
        [
            (.5,0,0),
            (0,0,0),
            (-.5,0,0)
        ],
        [
            (1,0,0),
            (0,0,0),
            (-1,0,0)
        ]
    ]

    for index in range(len(multiples)):
        result = scale(polygon, multiples[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def get_center(points):
    return tuple([np.average([x[index] for x in points]) for index in range(len(points[0]))])
    
def convert_point_to_line(origin, start, end, distance=1.0, count=2):

    A = move_point(origin, None, start=end, end=start, distance=distance/2.0)
    B = move_point(origin, None, start=start, end=end, distance=distance/2.0)
    line = [A, B]

    if count > 2:
        for index in range(count - 2):
            division = float(count - 1) / float(index + 1)
            new_point = get_point_on_points(line, division=division)
            line.insert(-1, new_point)

    return line

def normalize(vector):
    
    # 1. 벡터 길이 계산
    length = math.sqrt(reduce(lambda x, y : x + y, [ele**2 for ele in vector]))
    normal_vector = tuple([ele / length for ele in vector])

    # 2. Normalize
    try:
        normal_vector = tuple([ele / length for ele in vector])
    except ZeroDivisionError:
        normal_vector = vector

    return normal_vector

def normalize_test():
    
    vectors = [
        (1,0,0),
        (2,0,0),
        (-3,0,0),
        (0,4,0),
        (0,-5,0),
        (0,0,6),
        (1,1,0),
        (1,1,1)
    ]

    predicts = [
        (1,0,0),
        (1,0,0),
        (-1,0,0),
        (0,1,0),
        (0,-1,0),
        (0,0,1),
        (1.0/math.sqrt(2),1.0/math.sqrt(2),0),
        (1.0/math.sqrt(3), 1.0/math.sqrt(3), 1.0/math.sqrt(3))
    ]

    for index in range(len(vectors)):
        result = normalize(vectors[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def smooth_z(line, sigma=1.0):

    z = np.array([point[2] for point in line])

    z_smoothed = gaussian_filter1d(z, sigma=sigma)
    # smoothed_line = (
    #     line[:1] + 
    #     [(line[index][0], line[index][1], z_smoothed[index]) for index in range(len(line[1:-1]))] +
    #     line[-1:]
    # )
    smoothed_line = [(line[index][0], line[index][1], z_smoothed[index]) for index in range(len(line))]
    return smoothed_line

def smooth_z_test():
    
    lines = [
        [
            (0,0,0),
            (1,0,.5),
            (1.5,0,2),
            (2,0,.5),
            (3,0,1),
            (4,0,1.4),
            (4.5,0,2),
            (5,0,1.6),
            (5.7,0,3),
            (6,0,3.7),
            (6.5,0,4),
            (7,0,7)
        ],
    ]

    for index in range(len(lines)):
        x = np.array(range(len(lines[index])))
        plt.plot(x, np.array([point[2] for point in lines[index]]))
        plt.plot(x, np.array([point[2] for point in smooth_z(lines[index], sigma=2)]))
        # plt.plot(x, np.array([point[2] for point in smooth_z(lines[index], sigma=1.25)]))
        # plt.plot(x, np.array([point[2] for point in smooth_z(lines[index], sigma=.5)]))
        # plt.plot(x, np.array([point[2] for point in smooth_z(lines[index], sigma=.25)]))
        plt.title("Spline Curve Using the Gaussian Smoothing")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

def reduce_polygon(points, side_count):
    """
    다각형 변 개수 축소
    - 다각형의 좌표는 시작/종료가 중복되지 않는 것을 전제로 한다.
    """

    # - 삼각형 미만 설정 시 종료
    if len(points) < 4 or side_count < 3:
        return points

    # - 이미 다각형의 변 개수가 목표 개수(side_count) 이하인 경우 종료
    if len(points) <= side_count:
        return points

    # 1. 단순화 
    # - 미세하게 볼록한 부분을 제거
    points = simplify_polygon(points)

    # 2. Convex Hull 추출
    points = get_convex(points)

    # - 목표 변 개수에 도달할 때까지 반복
    while len(points) > side_count:
        # 1. 다각형의 연속한 3개 좌표에서 최소 수직거리를 추출
        min_dist = float("inf")
        min_index = -1
        for index in range(len(points)):
            A = points[index]
            B = points[(index + 1) % len(points)]
            C = points[(index + 2) % len(points)]
            dist = calc_distance_from_line([A, C], B)
            if dist < min_dist:
                min_dist = dist
                min_index = (index + 1) % len(points)

        # 3. 최소 수직거리에 해당하는 좌표 제거
        points.pop(min_index)

    # - 시작/종료 좌표가 중복되는 경우 제거
    if points[0] == points[-1]:
        points = points[:-1]

    return points

def reduce_polygon_test():

    points = [
        (0,0,0),
        (4,0,0),
        (4,2,0),
        (3,3,0),
        (0,2,0),
    ]

    side_count = 4

    result = reduce_polygon(points, side_count)
    log_print(result)

# [Quad] --------------------------------------------------

def create_quad_tree(points, unit=1.0):
    """
    좌표목록(points)에 기반한 quad tree 반환
    - quad 탐색(get_closest_quad_point) 사용을 위한 사전단계
    """

    def create_quad_map(points):

        # 1. 좌표목록의 최소/최대값 추출
        min_x = min([x[0] for x in points])
        max_x = max([x[0] for x in points])
        min_y = min([x[1] for x in points])
        max_y = max([x[1] for x in points])

        # 2. 좌측하단 좌표값 / 우측상단 좌표값 추출
        left_bottom = (
            int(min_x) if min_x > 0 else (min_x if min_x % 1 == 0.0 else int(min_x) - 1), 
            int(min_y)if min_y > 0 else (min_y if min_y % 1 == 0.0 else int(min_y) - 1))
        
        width = 1
        while left_bottom[0] + width < max_x or left_bottom[1] + width < max_y:
            width *= 2

        # 3. 전체 quad 맵의 중점을 구한다.
        center_point = [x + width / 2.0 for x in left_bottom]

        return center_point, width

    def create_node(center_point, width):
        return {
            "center" : center_point,
            "left_top" : (center_point[0] - width / 2.0, center_point[1] + width / 2.0),
            "right_top" : (center_point[0] + width / 2.0, center_point[1] + width / 2.0),
            "right_bottom" : (center_point[0] + width / 2.0, center_point[1] - width / 2.0),
            "left_bottom" : (center_point[0] - width / 2.0, center_point[1] - width / 2.0),
            "child_nodes" : [],
            "points" : [],
        }        

    def register_point(node, point):
        
        width = node["left_top"][1] - node["left_bottom"][1] 

        # 1. 단말에 도착한 경우
        if width <= unit:
            # - 좌표를 등록한다.
            node["points"].append(point)
            return

        # 2. width 가 최소단위(unit)보다 크고 자식 node 가 없다면 분할한다.
        if width > unit and len(node["child_nodes"]) < 1:
            left_top = create_node(get_mid(node["left_top"], node["center"]), width / 2.0)
            right_top = create_node(get_mid(node["right_top"], node["center"]), width / 2.0)
            right_bottom = create_node(get_mid(node["right_bottom"], node["center"]), width / 2.0)
            left_bottom = create_node(get_mid(node["left_bottom"], node["center"]), width / 2.0)
            node["child_nodes"] += [left_top, right_top, right_bottom, left_bottom]

        # 2. 해당하는 하위 node 로 단계 하강
        # - 1) 좌상단
        if point[0] <= node["center"][0] and point[1] >= node["center"][1]:
            register_point(node["child_nodes"][0], point)
        # - 2) 우상단
        elif point[0] >= node["center"][0] and point[1] >= node["center"][1]:
            register_point(node["child_nodes"][1], point)
        # - 3) 우하단
        elif point[0] >= node["center"][0] and point[1] <= node["center"][1]:
            register_point(node["child_nodes"][2], point)
        # - 4) 좌하단
        elif point[0] <= node["center"][0] and point[1] <= node["center"][1]:
            register_point(node["child_nodes"][3], point)

    if len(points) < 1:
        return None

    # 1. quad 범위 생성 (중점, 너비 추출)
    center_point, width = create_quad_map(points)

    # 2. root node 생성
    root_node = create_node(center_point, width)

    # 3. quad tree 생성
    [register_point(root_node, x) for x in points]

    return root_node

def calc_node_distance(node, point):
    """
    quad tree 에 속한 node 와 특정 좌표와의 최소거리를 반환
    """

    # node 의 내부에 속한 좌표의 경우 0.0 반환
    if check_inside(point, [node["left_top"], node["right_top"], node["right_bottom"], node["left_bottom"]]):
        return 0.0

    # 1. node 의 어느 방향에 좌표가 위치하는지 검사
    # - 1) 좌상단
    if point[0] <= node["center"][0] and point[1] >= node["center"][1]:
        if point[0] > node["left_top"][0]:
            return point[1] - node["left_top"][1]
        elif point[1] < node["left_top"][1]:
            return node["left_top"][0] - point[0]
        else:
            return calc_distance(node["left_top"], point)
    # - 2) 우상단
    elif point[0] >= node["center"][0] and point[1] >= node["center"][1]:
        if point[0] < node["right_top"][0]:
            return point[1] - node["right_top"][1]
        elif point[1] < node["right_top"][1]:
            return point[0] - node["right_top"][0]
        else:
            return calc_distance(node["right_top"], point)
    # - 3) 우하단
    elif point[0] >= node["center"][0] and point[1] <= node["center"][1]:
        if point[0] < node["right_bottom"][0]:
            return node["right_bottom"][1]- point[1]
        elif point[1] > node["right_bottom"][1]:
            return point[0] - node["right_bottom"][0]
        else:
            return calc_distance(node["right_bottom"], point)
    # - 4) 좌하단
    elif point[0] <= node["center"][0] and point[1] <= node["center"][1]:
        if point[1] > node["left_bottom"][1]:
            return node["left_bottom"][0] - point[0]
        elif point[0] > node["left_bottom"][0]:
            return node["left_bottom"][1] - point[1]
        else:
            return calc_distance(node["left_bottom"], point)

    # 2. 해당 node 의 꼭짓점과 target 의 수직거리를     

def get_closest_quad_point(target, root_node, except_points={}):

    def check_except(point):
        if except_points.get(point) != None:
            return True
        return False

    if None in [target, root_node]:
        log_print("오류")
        return None

    min_point = None
    min_dist = float("inf")

    node_stack = [root_node]

    while len(node_stack) > 0:
        # 1. stack 에서 최후미의 node 를 추출한다.
        curr_node = node_stack.pop(-1)

        # - 현재 node 와의 거리가 현재까지의 최단거리보다 먼 경우 생략
        if calc_node_distance(curr_node, target) > min_dist:
            continue

        # 2-1. 단말 node 가 아닌 경우 하위 node 로 진행
        if len(curr_node["child_nodes"]) > 0:
            child_nodes = sorted(curr_node["child_nodes"], key=lambda x : calc_node_distance(x, target), reverse=True)
            node_stack += child_nodes
        # 2-2. 단말 node 인 경우 
        else:
            # 3-1. 좌표가 존재하는 경우 최단거리를 비교한다.
            if len(curr_node["points"]) > 0:
                for node_point in curr_node["points"]:
                    #* 특정 좌표 제외 추가
                    if not check_except(node_point):
                        dist = calc_distance(node_point, target)                
                        if dist < min_dist:
                            min_point = node_point
                            min_dist = dist

    return min_point

def quad_test():

    points = [(x, y, 0) for x in range(11) for y in range(11)]

    # 1. quad tree 생성
    root_node = create_quad_tree(points)

    # 2. 검색
    for x in range(11):
        for y in range(11): 
            result = get_closest_quad_point((x + .4, y + .4, 0), root_node)
            if result != (x, y, 0):
                log_print("Index : {0} - {1}".format(x, y))
                log_print("Predict : {0}".format((x, y, 0)))
                log_print("Result : {0}".format(result))
                break
        if y < 10:
            break
    else:
        log_print("Test Complete [OK]")   

# [교차] --------------------------------------------------

def intersection(p1, p2, p3, p4, alt_index=0):
    """
    2차원 평면 상 2 개의 직선(p1-p2 / p3-p4) 간 교차점 반환
    - 평행한 경우 None 반환
    """

    a1 = float(p2[1] - p1[1])
    b1 = float(p1[0] - p2[0])
    c1 = float(a1 * p1[0] + b1 * p1[1])
    a2 = float(p4[1] - p3[1])
    b2 = float(p3[0] - p4[0])
    c2 = float(a2 * p3[0] + b2 * p3[1])

    determinant = float(a1 * b2 - a2 * b1)
    # - 평행한 경우
    if determinant == 0.0:
        return None
    # - 평행하지 않은 경우
    else:
        # x, y 값 추출
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        if alt_index == 0:
            z = calc_z(p1, p2, (x, y))        
        else:
            z = calc_z(p3, p4, (x, y))

        return (x, y, z)

def intersection_test():

    p1 = (0,0,0)
    p2 = (10,0,0)

    lines = [
        [(0,1,0), (0,-1,-2)],
        [(0,5,-10), (10,-5,-10)],
        [(0,-5,-10), (10,5,0)]
    ]

    predicts = [
        (0, 0, 0), (0, 0, -1), 
        (5, 0, 0), (5, 0, -10),
        (5, 0, 0), (5, 0, -5)
    ]

    for index in range(len(lines)):
        result = intersection(p1, p2, lines[index][0], lines[index][1])
        log_print("Result = {0} / {1}".format(result, predicts[index * 2]))
        if result != predicts[index * 2]:
            log_print("Diff")
            break
        result = intersection(p1, p2, lines[index][0], lines[index][1], alt_index=1)
        log_print("Result = {0} / {1}".format(result, predicts[index * 2 + 1]))
        if result != predicts[index * 2 + 1]:
            log_print("Diff")
            break
    else:
        log_print("Test Complete [OK]")

def line_intersection(first, second):
    """
    2 개의 직선(first, second) 간 교차점 반환
    - 평행한 경우 None 반환
    """
    return intersection(first[0], first[-1], second[0], second[-1])

def check_intersection(p1, p2, p3, p4):
    """
    2개의 직선(p1-p2 / p3-p4)의 교차여부 반환
    """
    def ccw(p1, p2, p3):
        return np.sign((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

    try:
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    except KeyboardInterrupt:
        raise
    except:
        #* 세개의 좌표가 1열로 배열되며, 그중 1개의 좌표가 float("inf") 에 해당하는 경우 numpy.FloatingPointError 발생 
        #* 예) (0,0,0) (1,0,0) (float("inf"), 0, 0) => nan 반환
        log_print("Exception : {0}".format((p1, p2, p3, p4)))

def check_intersection_test():
    
    line_pairs = [
        (
            [(0,0,0), (10,0,0)],
            [(0,1,0), (10,1,0)]
        ),
        (
            [(0,0,0), (10,0,0)],
            [(5,-5,0), (5,5,0)]
        ),
        (
            [(0,0,0), (10,0,0)],
            [(0,-5,0), (0,5,0)]
        ),
        (
            [(0,0,0), (10,0,0)],
            [(10,-5,0), (10,5,0)]
        ),
        (
            [(0,0,0), (10,0,0)],
            [(-1,-5,0), (-1,5,0)]
        ),
        (
            [(0,0,0), (10,0,0)],
            [(11,-5,0), (11,5,0)]
        ),
        (
            [(0,0,0), (float('inf'),0,0)],
            [(-5,-5,0), (-5,5,0)]
        ),
        (
            [(0,0,0), (float("inf"),0,0)],
            [(15,-5,0), (15,5,0)]
        ),
    ]

    predicts = [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
    ]

    for index in range(len(line_pairs)):
        result = check_intersection(line_pairs[index][0][0], line_pairs[index][0][1], line_pairs[index][1][0], line_pairs[index][1][1])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

def check_intersection_on_points(line, points):
    """
    선(line)과 좌표목록(points)을 구성하는 선분의 교차여부 반환
    """

    for seg in [points[index:index+2] for index in range(len(points)-1)]:
        intersect_p = intersection(seg[0], seg[-1], line[0], line[-1])
        if intersect_p != None:
            if check_point_on_line(seg, intersect_p) and check_point_on_line(line, intersect_p):
                return True
    else:
        return False

def check_point_on_line(line, point, floating_error=0.001, is_debug=False):
    """
    선분(line)상에 좌표(point)의 위치 여부 반환
    """    

    # - 소수점 반올림
    _point = tuple([round(x, 5) for x in point])    

    # 1. 선 길이 추출
    line_length = calc_distance(line[0], line[-1])
    
    # 2. 좌표와 선 시작/종료 사이의 거리 합 추출
    point_length = calc_distance(_point, line[0]) + calc_distance(_point, line[-1])

    if is_debug:
        log_print("Diff : {0}".format(round(point_length - line_length, 5)))

    # - 거리 차이 계산 시 부동소수점 오차 고려
    if abs(round(line_length - point_length, 5)) < floating_error:
        return True

    return False

def check_point_on_line_test():

    line = [
        (0,0,0),
        (10,2,0)
    ]

    points = [
        (0,0,0),
        (10,0,0),
        (5,1,0),
        (2.5, .5, 0),
        (7.5, 1.5, 0)
    ]

    predicts = [
        True,
        True,
        True,
        True,
        True
    ]

    for index in range(len(points)):
        result = check_point_on_line(line, points[index])
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")
    
def check_curve_intersect(first, second):
    """
    2개 곡선의 교차여부 반환
    """

    for seg_a in [first[i:i+2] for i in range(len(first)-1)]:
        for seg_b in [second[j:j+2] for j in range(len(second)-1)]:
            if check_intersection(seg_a[0], seg_a[-1], seg_b[0], seg_b[-1]):
                return True
    return False

def get_intersection_on_points(line, points):
    """
    선(line)과 좌표목록(points)을 구성하는 선분의 교차점 1개를 반환한다.
    - 교차하는 선분이 없는경우 None 반환
    - 교차하는 선분이 1개인 경우 그대로 반환
    - 교차하는 선분이 2개 이상인 경우, line 의 중점(get_mid())에 가장 가까운 교차점을 반환
    """

    intersect_points = []

    for segment in [points[index:index+2] for index in range(len(points) - 1)]:
        intersect_point = intersection(line[0], line[-1], segment[0], segment[-1])
        if intersect_point != None:
            if check_point_on_line(segment, intersect_point) and check_point_on_line(line, intersect_point):
                intersect_points.append(intersect_point)

    if len(intersect_points) < 1:
        return None

    if len(intersect_points) < 2:
        return intersect_points[0]

    return sorted(intersect_points, key=lambda x : calc_distance(x, get_mid(line[0], line[-1])))[0]

def get_intersection_on_points_test():

    points = [
        (0,0,0), (5,0,0), (10,0,0)
    ]

    lines = [
        [(0,-5,0),(0,5,0)],
        [(5,-5,0),(5,5,0)],
        [(10,-5,0),(10,5,0)],
        [(1,-5,0),(1,5,0)],
        [(6,-5,0),(6,5,0)],
        [(0,-5,0),(5,5,0)],
        [(0,1,0),(0,5,0)],
    ]

    predicts = [
        (0,0,0),
        (5,0,0),
        (10,0,0),
        (1,0,0),
        (6,0,0),
        (2.5,0,0),
        None,
    ]

    for index in range(len(lines)):
        result = get_intersection_on_points(lines[index], points)
        if result != predicts[index]:
            log_print("Index : {0}".format(index+1))
            log_print("Predict : {0}".format(predicts[index]))
            log_print("Result : {0}".format(result))
            break
    else:
        log_print("Test Complete [OK]")

# os.system("clear")
# check_inside_test()

# post_A = (451653.4230, 3950761.0690, 57.4900)
# post_B = (451689.7215, 3950781.9736, 58.9490)

# post_C = move_point(post_A, None, start=post_A, end=post_B, distance=19.5)
# post_D = move_point(post_A, None, start=post_A, end=post_B, distance=19.5 + 3.75)

# line = get_ortho_line(post_D, start=post_A, end=post_B, dist1=4.33, dist2=-(4.33 + 4.5))
# light_A, light_B = line[0], line[-1]
# light_A = move_point(light_A, None, start=post_B, end=post_A, distance=.2)
# light_B = move_point(light_B, None, start=post_B, end=post_A, distance=.2)

# light_C = get_ortho_line(post_B, start=post_A, end=post_B, dist1=6.87, dist2=6.87)[-1]
# light_B = move_point(light_B, None, start=post_B, end=post_A, distance=.2)

# plane_A = [
#     (451667.6900, 3950779.5790, 58.2835),
#     (451669.8820, 3950780.8560, 58.3800),
#     (451669.4695, 3950777.0115, 58.1865),
# ]
# plane_B = [
#     (451669.4695, 3950777.0115, 58.1865),
#     (451671.6440, 3950778.2610, 58.2720),
#     (451672.0990, 3950774.9550, 58.1210),
# ]
# plane_C = [
#     (451685.3810, 3950786.1780, 58.8300),
#     (451687.8710, 3950787.6110, 58.9490),
#     (451685.1450, 3950789.6360, 58.9970),
# ]

# alt_A = calc_z_from_plane(plane_A, light_A[0], light_A[1])
# alt_B = calc_z_from_plane(plane_B, light_B[0], light_B[1])
# alt_C = calc_z_from_plane(plane_C, light_C[0], light_C[1])

# print(light_A)

# print(alt_A + 5.59)
# print(alt_B + 5.59)
# print(alt_C + 6.07)

# arrow_point = (450946.8947, 3950234.9166, 45.5889)
# A2 = [
#     (450952.5220, 3950241.3680, 45.3270), 
#     (450952.5400, 3950266.4010, 45.0490)
# ]
# stop_point = move_point(arrow_point, None, start=A2[0], end=A2[-1], distance=31.49)

# post_A = (450955.6030, 3950270.7910, 45.3470)
# post_B = move_point(post_A, None, start=A2[0], end=A2[-1], distance=16.94)
# print(post_B)

# line = get_ortho_line(post_A, start=A2[0], end=A2[-1], dist1=4.36, dist2=-(4.36 + 4.58))
# light_A = line[-1]
# light_B = line[0]

# light_A = move_point(light_A, None, start=A2[-1], end=A2[0], distance=0.3)
# light_B = move_point(light_B, None, start=A2[-1], end=A2[0], distance=0.3)

# light_C = get_ortho_line(post_B, start=A2[0], end=A2[-1], dist1=6.36, dist2=6.36)[-1]
# light_C = move_point(light_C, None, start=A2[-1], end=A2[0], distance=0.3)
# print(light_C)

# plane_A = [
#     (450944.5150, 3950268.0580, 45.1690),
#     (450954.3060, 3950269.0090, 44.9960),
#     (450949.3330, 3950278.8860, 44.9630),
# ]
# plane_B = [
#     (450944.5150, 3950268.0580, 45.1690),
#     (450954.3060, 3950269.0090, 44.9960),
#     (450949.3330, 3950278.8860, 44.9630),
# ]
# plane_C = [
#     (450947.6990, 3950288.0480, 44.8950),
#     (450950.9480, 3950288.1140, 44.8350),
#     (450949.3610, 3950285.6030, 44.8930),
# ]

# alt_A = calc_z_from_plane(plane_A, light_A[0], light_A[1])
# alt_B = calc_z_from_plane(plane_B, light_B[0], light_B[1])
# alt_C = calc_z_from_plane(plane_C, light_C[0], light_C[1])
# # print(alt_A + 6.135)
# # print(alt_B + 6.135)
# print(alt_C + 6.235)

A = (450946.0470, 3950288.0475, 0)
B = (450946.0530, 3950290.5880, 0)
C = (450949.2550, 3950287.4356, 0)
seg = [A, B]

ortho = get_ortho_line(C, start=A, end=B, dist1=-0.2, dist2=-0.2)
print(ortho)
