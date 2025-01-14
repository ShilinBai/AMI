import functools
import heapq
import os
import re
import shlex
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import pandas as pd
import time
import bisect
from ami_mask import AMI_Mask
from find_zone import find_zone
from calculate import *

COLOR_MIN = 5e-4


class DataSourceHandler:
    @staticmethod
    def center_coordinates(scatter_point):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        x_max = max(scatter_point, key=lambda lis: lis[0])
        color_max = max(scatter_point, key=lambda lis: lis[-1])
        cycle_counts = [tp[-1] for tp in scatter_point].count(color_max[-1])
        color_max_counts = heapq.nlargest(cycle_counts, scatter_point, key=lambda lis: lis[-1])
        if len(color_max_counts) < 2:
            print("Center Failed!Does not contain data for multiple cycles!")
            return scatter_point

        color_max_counts.sort(key=lambda lis: lis[0], reverse=True)
        target_value = find_nearest([tp[1] for tp in scatter_point], 0)
        all_target_index = [x for (x, y) in enumerate([tp[1] for tp in scatter_point]) if y == target_value]
        voltage_0 = [scatter_point[i] for i in all_target_index]
        voltage_max = heapq.nlargest(cycle_counts, voltage_0, key=lambda lis: lis[-1])
        x_middle = functools.reduce(lambda x, y: x + y, [tp[0] for tp in voltage_max]) / len(voltage_max)
        delta_x = x_max[0] / 2 - x_middle
        scatter_point_new = []
        for point in scatter_point:
            x_new = point[0] + delta_x
            if x_new < x_max[0]:
                scatter_point_new.append((point[0] + delta_x, point[1], point[2]))
            else:
                scatter_point_new.append((point[0] + delta_x - x_max[0], point[1], point[2]))

        return scatter_point_new

    @staticmethod
    def handle_hspice_eye_diagram_data(data_map_path, center):
        data = []
        with open(data_map_path, "r") as fr:
            line_data = []
            for line in fr:
                if line == "\n" or line == "":
                    data.append(line_data)
                    line_data = []
                else:
                    data_info = line.split(" ")
                    line_data.append((float(data_info[0]) * 10 ** 12, float(data_info[1]), float(data_info[2]) * 10))

        scatter_point = []
        for line in data:
            scatter_point.extend(i for i in line)

        if center:
            return DataSourceHandler.center_coordinates(scatter_point)
        return scatter_point

    @staticmethod
    def get_hspice_eye_diagram_height_and_width_from_lis_file(lis_file):
        height, height_time, width = None, None, None
        with open(lis_file, "r") as fr:
            for line in fr:
                if line.startswith("Maximum eye height"):
                    lis = shlex.split(line)
                    height = lis[-1]
                    for st in lis:
                        if st.startswith("t="):
                            height_time = st.split(",")
                            height_time = height_time[0][height_time[0].index("=") + 1:]
                if line.startswith("Maximum eye width"):
                    width = line.strip().split(" ")[-1]
        return height, height_time, width

    @staticmethod
    def handle_ads_eye_diagram_data(spectra_raw_path, center):
        data = []
        add_value = False
        with open(spectra_raw_path, "r") as fr:
            point_data = []
            for line in fr:
                if line == "Plotname: ChannelSim1[1].TDM.EyeMeasurements.Eye_Probe1\n":
                    break
                if line.startswith("Sweep Variables:"):
                    color_num = line.split()[-1]
                    if float(color_num) > 100:
                        break

                if line.startswith("Values:"):
                    add_value = True
                    continue

                if line == '#\n':
                    add_value = False
                    data.append(point_data)
                    point_data = []
                    continue

                if add_value:
                    data_info = line.split()
                    if not data_info:
                        continue
                    point_data.append((float(data_info[1]) * 10 ** 12, float(data_info[2]), float(color_num)))

        scatter_point = []
        for line in data:
            scatter_point.extend(i for i in line)

        if center:
            # todo
            pass
        return scatter_point


class EyeDiagramPainter:
    def __init__(self, path, save, source='HSPICE', width=10, height=6, cmap='HSPICE', background='white',
                 center=False,data_rate="6400"):
        """
        Args:
            source: ads/hspice,
            path: data source path,
            save: export image path,
            width: image width,
            height: image height
            cmap: color bar style {ADS, HSPICE, gnuplot ...},
            background: background color {None, black, white}
            center : center eye diagram {False, True}
        """
        self.source = source
        self.image_path = save
        self.data_rate = data_rate
        self.data = self._get_data(path, center)

        self.cmap = cmap
        self.width = width
        self.height = height
        self.background = background
        self.mask = self._find_mask()


    def _data_pre_process(self):

        use_zone = self._find_use_zone()
        gap_pair_x = self._find_max_x_long(use_zone)
        pair_max_x = max(gap_pair_x, key=lambda item: item[0])
        x_start = pair_max_x[1][0][0]
        x_end = pair_max_x[1][1][0]
        return list(filter(lambda item: item[0] >= x_start*0.9 and item[0] < x_end*1.1, self.data))


    def _find_mask(self):
        if not self.data:
            return []

        tdivw1,tdivw2,vdivw = AMI_Mask("LPDDR5",str(int(self.data_rate)//2)).get_mask(ui=False)

        pre_data = self._data_pre_process()
        outline_zone = self._find_outline(pre_data)
        gap_pair_x = self._find_max_x_long(outline_zone)
        gap_pair_y = self._find_max_y_long(outline_zone)
    

        # y_middle = max(gap_pair_x, key=lambda item: item[0])[1][0][1]
        # x_middle = max(gap_pair_y, key=lambda item: item[0])[1][0][0]
        # point1 = [item[1][0] for item in gap_pair_x] + [item[1][1] for item in gap_pair_x]
        # point2 = [item[1][0] for item in gap_pair_y] + [item[1][1] for item in gap_pair_y]
        # return {"mask" : [],
        #         "outline_data":outline_zone,
        #         "outline" : [[item[0] for item in point1+point2],[item[1] for item in point1+point2]]}


        x_pair = max(gap_pair_x, key=lambda item: item[0])[1]
        x_middle,y_middle = (x_pair[0][0] + x_pair[1][0])/2,x_pair[0][1]


        #mask
        points = {
            "mask_center":[x_middle,y_middle],
            "left_middle":[x_middle - tdivw1*1e12/2,y_middle],
            "left_top":[x_middle - tdivw2*1e12/2,y_middle + vdivw/2],
            "right_top":[x_middle + tdivw2*1e12/2,y_middle + vdivw/2],
            "right_middle":[x_middle + tdivw1*1e12/2,y_middle],
            "right_bottom":[x_middle + tdivw2*1e12/2,y_middle - vdivw/2],
            "left_bottom":[x_middle - tdivw2*1e12/2,y_middle - vdivw/2],
        }

        # Check if the mask is located within the edge
        # need_to_train = self._judge_train(points,outline_zone)

        x_middle,y_middle = self._adjust_mask(gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw)
        mask = []

        # margin

        margin = self._get_margin(x_middle,y_middle,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw)

        # jitter
        jitter = self._get_jitter(x_middle,y_middle,margin["eye_width"])
        margin["jitter"] = jitter


        mask.append([
            x_middle - tdivw1*1e12/2,
            x_middle - tdivw2*1e12/2,
            x_middle + tdivw2*1e12/2,
            x_middle + tdivw1*1e12/2,
            x_middle + tdivw2*1e12/2,
            x_middle - tdivw2*1e12/2,
        ])
        mask.append([
            y_middle,
            y_middle + vdivw/2,
            y_middle + vdivw/2,
            y_middle,
            y_middle - vdivw/2,
            y_middle - vdivw/2,
        ])


        return {"mask" : mask,
                "outline_data":outline_zone,
                "outline" : [[item[0] for item in outline_zone],[item[1] for item in outline_zone]],
                "margin" : margin}


    def _judge_train(self,points,outline_zone):

        center = points["mask_center"]
        left_middle = points["left_middle"]
        left_top = points["left_top"]
        right_top = points["right_top"]
        right_middle = points["right_middle"]
        right_bottom = points["right_bottom"]
        left_bottom = points["left_bottom"]

        zone1 = list(filter(lambda item: item[0] <= left_top[0], outline_zone))
        zone2 = list(filter(lambda item: item[0] > right_top[0], outline_zone))
        zone3 = list(set(outline_zone) - set(zone1) - set(zone2))

        
        
        return center[0],center[1]
    
    def _adjust_mask(self,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw):
        x_pair = max(gap_pair_x, key=lambda item: item[0])[1]
        x_middle,y_middle = (x_pair[0][0] + x_pair[1][0])/2,x_pair[0][1]
        # offset_x = calculate_offest_x(x_middle,y_middle,gap_pair_x,tdivw1,tdivw2,vdivw)
        # offset_y = calculate_offest_y(x_middle,y_middle,gap_pair_y,tdivw1,tdivw2,vdivw)
        x_middle,y_middle = calculate_center(x_middle,y_middle,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw)

        return x_middle,y_middle

    def _get_margin(self,x_middle,y_middle,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw):

        eye_width = max(gap_pair_x, key=lambda item: item[0])[0]

        eye_hight = max(gap_pair_y, key=lambda item: item[0])[0]


        setup,hold = calculate_set_up_and_hold_by_judge_value(
            y_middle + vdivw/2,
            sorted([item[1] for item in gap_pair_x],key=lambda item: item[0][1]),
            (x_middle- tdivw2*1e12/2),
            (x_middle + tdivw2*1e12/2),
            True)
        top,bottom = calculate_top_and_bottom_by_judge_value(
            x_middle + tdivw2*1e12/2,
            sorted([item[1] for item in gap_pair_y],key=lambda item: item[0][0]),
            y_middle - vdivw/2,
            y_middle + vdivw/2,
            True)
        # c = top+bottom+vdivw
        margin = {
            "eye_hight":top+bottom+vdivw if top and bottom else eye_hight,
            "eye_width":eye_width,
            "top_margin":top,
            "bottom_margin":bottom,
            "setup_margin":setup,
            "hold_margin":hold
        }

        return margin


    def _get_jitter(self,x_middle,y_middle,x_scale):
        use_zone = list(filter(lambda item: item[1] >= y_middle*0.99 and item[1] < y_middle*1.01, list(filter(lambda item: item[0] < x_middle and item[0] >= x_middle - x_scale and item[2] >= COLOR_MIN, self.data))))
        y_list = list(set([item[1] for item in use_zone]))
        y_list.sort()
        if y_middle in y_list:
            # x_points = list(filter(lambda item: item[1] == y_middle, use_zone))
            # x_list = [item[0] for item in x_points]
            # return max(x_list) - min(x_list)
            y_value = y_middle
        else:
            try:
                y_index = bisect.bisect_left(y_list,y_middle)
                y_value = y_list[y_index]
            except Exception:
                return 0
        x_points = list(filter(lambda item: item[1] == y_value, use_zone))
        x_list = [item[0] for item in x_points]
        return max(x_list) - min(x_list)


    def _find_outline(self,pre_data):

        x,y,c = self._get_scatter_points(pre_data)
        x,y = list(set(x)),list(set(y))
        x.sort()
        y.sort()
        c_value = COLOR_MIN
        if self.source == 'AMI':
            c = [point[2] for point in list(filter(lambda item: item[2] > 0, self.data))]
            c_value = min(c) + (max(c) - min(c))* 0.01
        # x_scale,y_scale = (max(x)-min(x))/len(x)*1.1,(max(y)-min(y))/len(y)*1.1
        x_scale,y_scale = min(np.abs(np.array(x)[1:] - np.array(x)[:-1])),min(np.abs(np.array(y)[1:] - np.array(y)[:-1]))
        distance = x_scale**2 + y_scale**2
        x = 0.5 * (max(x) + min(x))
        # y = 0.5 * (y_lim[0] + y_lim[1])
        y = 0.5 * (max(y) + min(y))
        filtered_data = list(filter(lambda item: item[2] > c_value, pre_data))

        left_up = list(filter(lambda item: item[0] <= x and item[1] > y, filtered_data))
        left_down = list(filter(lambda item: item[0] <= x and item[1] <= y, filtered_data))
        right_up = list(filter(lambda item: item[0] > x and item[1] > y, filtered_data))
        right_down = list(filter(lambda item: item[0] > x and item[1] <= y, filtered_data))
        
        left_up_line = []
        left_down_line = []
        right_up_line = []
        right_down_line = []

        left_up_line = self._find_line_in_zone(left_up,-1,1,distance,x_scale,y_scale)
        left_down_line = self._find_line_in_zone(left_down,-1,-1,distance,x_scale,y_scale)
        right_up_line = self._find_line_in_zone(right_up,1,1,distance,x_scale,y_scale)
        right_down_line = self._find_line_in_zone(right_down,1,-1,distance,x_scale,y_scale)

        all_line = []
        all_line.extend(left_up_line)
        all_line.extend(left_down_line)
        all_line.extend(right_up_line)
        all_line.extend(right_down_line)
        return all_line
        # outline = []
        # for line in [right_up_line,right_down_line,left_up_line,left_down_line]:
        #     outline = splicing_line_segments(outline,line)
        
        # return outline



    def _find_outline_use_find_zone(self):
        x,y,c = self._get_scatter_points()
        c_value_dict = self._get_c_value_dict()

        x = list(set(x))
        y = list(set(y))
        matrix = np.zeros((len(x),len(y)))
        for index_x,x_value in enumerate(x):
            for index_y,y_value in enumerate(y):
                matrix[index_x][index_y] = c_value_dict[y_value][x_value]
        
        outline_zone = find_zone(matrix)

        points = []
        for pair in outline_zone:
            for point_index in pair:
                x_index = point_index[0]
                y_index = point_index[1]
                c_value = matrix[x_index][y_index]
                points.append((x[x_index],y[y_index],c_value))

        return points

    def _get_c_value_dict(self):
        c_value_dict = {}
        for point in self.data:
            x = point[0]
            y = point[1]
            c = point[2]
            if y in c_value_dict:
                c_value_dict[y][x] = c
            else:
                c_value_dict[y] = {
                    x:c
                }
        return c_value_dict

    def _find_line_in_zone(self,zone,derication_x,derication_y,distance,x_scale,y_scale):
        data_dict_x = {}
        data_dict_y = {}
        for point in zone:
            x = point[0]
            y = point[1]
            if y in data_dict_y:
                data_dict_y[y].append(point)
            else:
                data_dict_y[y] = [point]
    
            if x in data_dict_x:
                data_dict_x[x].append(point)
            else:
                data_dict_x[x] = [point]

        x_list = list(data_dict_x.keys())
        x_list.sort()
        y_list = list(data_dict_y.keys())

        points_y = []
        
        if derication_y < 0:
            y_list = y_list[::-1]
        for y_value in y_list:
            if derication_x < 0:
                point = max(data_dict_y[y_value], key=lambda item: item[0])
            else:
                point = min(data_dict_y[y_value], key=lambda item: item[0])
            points_y.append(point)

        len_y = len(points_y)
        
        boundary_x = min(points_y[:len_y//3],key=lambda item: item[0])[0] if derication_x < 0 else max(points_y[:len_y//3],key=lambda item: item[0])[0]
        x_index = x_list.index(boundary_x)
        x_list = x_list[x_index:] if derication_x < 0 else x_list[:x_index]

        points_x = []
        for x_value in x_list:
            if derication_y < 0:
                point = max(data_dict_x[x_value], key=lambda item: item[1])
            else:
                point = min(data_dict_x[x_value], key=lambda item: item[1])
            points_x.append(point)

        
        x_middle = x_list[-1] if derication_x < 0 else x_list[0]
        p1 = list(filter(lambda item: item[0] == x_middle, points_y))
        points_y = list(set(points_y) - set(p1))

        y_middle = y_list[0]
        p2 = list(filter(lambda item: item[1] == y_middle, points_x))
        points_x = list(set(points_x) - set(p2))

        # return points_x+points_y


        
        def filter_finally(all_points,p_judge = None):
            points = []
            if not all_points:
                return points
            p_judge = p_judge if p_judge else all_points[0]
            for point in all_points:
                x,y = point[0],point[1]
                x_judge,y_judge = p_judge[0],p_judge[1]
                x_distance = abs(x-x_judge)
                y_distance = abs(y_judge-y)
                if x_distance<=3 * x_scale and y_distance <= 4 * y_scale:
                    points.append(point)
                    p_judge = point
                # elif  (x-x_judge)**2 + (y_judge-y)**2 <= distance :
                #     points.append(point)
                #     p_judge = point
                else:

                    continue
            if len(points) < 5:
                points = []

            return points
        
        def filter_finally2(all_points):
            p1 = filter_finally(sorted(all_points, key=lambda point: (-point[0], -point[1])))
            p2 = filter_finally(sorted(all_points, key=lambda point: (-point[0], point[1])))
            p3 = filter_finally(sorted(all_points, key=lambda point: (point[0], point[1])))
            p4 = filter_finally(sorted(all_points, key=lambda point: (point[0], -point[1])))
            p5 = filter_finally(sorted(all_points, key=lambda point: (-point[1], -point[0])))
            p6 = filter_finally(sorted(all_points, key=lambda point: (-point[1], point[0])))
            p7 = filter_finally(sorted(all_points, key=lambda point: (point[1], point[0])))
            p8 = filter_finally(sorted(all_points, key=lambda point: (point[1], -point[0])))
            p = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
            # # p = []
            # # p = [p1,p2,p3,p4,p5,p6,p7,p8]
            # # p_len = [len(p1),len(p2),len(p3),len(p4),len(p5),len(p6),len(p7),len(p8)]
            # # p = p[p_len.index(max(p_len))]
            # # p.extend(p1)
            # # p.extend(p2)
            # # p.extend(p3)
            # # p.extend(p4)
            # # p.extend(p5)
            # # p.extend(p6)
            # # p.extend(p7)
            # # p.extend(p8)
            # # return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
            # p = p1
            # for line in [p2,p3,p4,p5,p6,p7,p8]:
            #     p = splicing_line_segments(p,line)
            return p

        points = []
        all_points = points_x+points_y
        cycle = 0
        while True:
            # if cycle == 1:
            #     break
            cycle_points = filter_finally2(all_points)
            # points = splicing_line_segments(points,cycle_points)
            points.extend(cycle_points)
            if cycle_points == []:
                break
            all_points = list(set(all_points) - set(cycle_points))
            cycle += 1

        
        # all_points = list(set(points))
        # if derication_x > 0 and derication_y > 0:
        #     points = sorted(all_points, key=lambda point: (point[0], -point[1]))
        # elif derication_x > 0 and derication_y < 0:
        #     points = sorted(all_points, key=lambda point: (-point[0], -point[1]))
        # elif derication_x < 0 and derication_y < 0:
        #     points = sorted(all_points, key=lambda point: (-point[0], point[1]))
        # elif derication_x < 0 and derication_y > 0:
        #     points = sorted(all_points, key=lambda point: (point[0], point[1]))
            



        
        return points
        # return remove_duplicates(points)
        # return list(set(all_points) - set(cycle_points))
    
        


    def _find_use_zone(self):
        if not self.data:
            return []

        x, y, c = self._get_scatter_points()

        filtered_data = list(filter(lambda item: item[2] > COLOR_MIN, self.data))
        y_min, y_max = np.min([value[1] for value in filtered_data]), np.max([value[1] for value in filtered_data])


        middle_y = 0.5 * (y_min + y_max)
        y_scale = 0.05 * (y_max - y_min)
        start_y = bisect.bisect_right(y, middle_y - y_scale)
        end_y = bisect.bisect_left(y, middle_y + y_scale)

        range_used = list(filter(lambda item: item[2] > 0, self.data[start_y:end_y]))

        return range_used

    def _find_max_x_long(self,zone):
        data_dict = {}
        for point in zone:
            y = point[1]
            if y in data_dict:
                data_dict[y].append(point)
            else:
                data_dict[y] = [point]
        
        gap_pair = []
        for key_y in data_dict:
            points = data_dict[key_y]
            points = sorted(points, key=lambda point: point[0])
            pair = max_gap(points)
            if pair:
                gap_pair.append(pair)
        
        return gap_pair

    def _find_max_y_long(self,zone):
        data_dict = {}
        for point in zone:
            x = point[0]
            if x in data_dict:
                data_dict[x].append(point)
            else:
                data_dict[x] = [point]
        
        gap_pair = []
        for key_x in data_dict:
            points = data_dict[key_x]
            points = sorted(points, key=lambda point: point[1])
            pair = max_gap(points,True)
            if pair:
                gap_pair.append(pair)
        
        return gap_pair


    def _get_data(self, path, center):
        # return:  data: {list {tuple}}: [(x0,y0,color),(x1,y1,color)...]
        if self.source == "HSPICE":
            return DataSourceHandler.handle_hspice_eye_diagram_data(path, center)
        elif self.source == "ADS":
            return DataSourceHandler.handle_ads_eye_diagram_data(path, center)
        elif self.source == "AMI":
            return self._get_data_csv(path)
        else:
            raise AttributeError

    def _get_data_csv(self, path):
        df = pd.read_csv(path)
        ui_t = (1/(float(self.data_rate) * 1e6)) * 1e12
        x_list = np.array(df['__UnitInterval []']) * ui_t
        # y_list = np.array(df['__Amplitude [V]']) * 1e3
        y_list = np.array(df['__Amplitude [V]'])
        c_list = np.array(df['EyeAfterProbe<b_input_22.int_ami_rx> []'])
        data = []
        for ind_x,x in enumerate(x_list):
            data.append((x,y_list[ind_x],c_list[ind_x]))
        return data

    def _get_scatter_points(self,data=None):
        data = self.data if data is None else data
        x, y, c = [], [], []
        for point in data:
            x.append(point[0])
            y.append(point[1])
            c.append(point[2])
        return np.array(x), np.array(y), np.array(c)

    def set_cmap_and_background(self, ax):
        if self.cmap == "ADS":
            # create color map line
            colormap = cm.jet(np.linspace(0.24, 0.64, 100))
            colormap[:1, :] = np.array([0, 0, 1, 1])
            colormap[:2, :] = np.array([0, 0, 0.8, 1])
            self.cmap = ListedColormap(colormap)
            ax.patch.set_facecolor(self.background)
        elif self.cmap == "HSPICE":
            colormap = cm.jet(np.linspace(0, 1, 256))
            # hspice: background is color point
            if self.background == "black":
                colormap[:1, :] = np.array([0, 0, 0, 1])
            elif self.background == "white":
                colormap[:1, :] = np.array([1, 1, 1, 1])
                # colormap[:1, :] = np.array([250/256, 250/256, 250/256, 1])
            self.cmap = ListedColormap(colormap)
        else:
            if self.source == "HSPICE":
                # cmap = cm.get_cmap(self.cmap).copy()
                cmap = matplotlib.colormaps[self.cmap]
                colormap = cmap(np.linspace(0, 1, 256))
                if self.background == "black":
                    colormap[:1, :] = np.array([0, 0, 0, 1])
                elif self.background == "white":
                    colormap[:1, :] = np.array([1, 1, 1, 1])
                self.cmap = ListedColormap(colormap)
            ax.patch.set_facecolor(self.background)

    def paint_scatter(self):
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        x, y, c = self._get_scatter_points()
        # c = normalize_c(c)
        # set color bar ande background
        self.set_cmap_and_background(ax)
        pc = ax.scatter(x, y, s=1, c=c, cmap=self.cmap)
        # paint color bar
        cb = fig.colorbar(pc, ax=ax)

        # def format_func(value, tick_number):
        #     return f"{round(value, 1)}m"

        # cb.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        # cb.set_label("Density", loc="top")
        # axis
        plt.gca().set_xlim([min(x), max(x)])
        plt.gca().set_ylim([min(y), max(y)])
        plt.xlabel("Time (ps)")
        plt.ylabel("Voltage (V)")
        # ax.axhline(y=0.11, color='red', linewidth=2, label='Red line')

        # mask
        if self.mask:
            mask = self.mask["mask"]
            outline = self.mask["outline"]
            # plt.scatter(x, y, c="red", cmap=self.cmap,alpha=0.5)
            # plt.scatter([item[0] for item in list(filter(lambda item: item[2] > 0, self.data))], [item[1] for item in list(filter(lambda item: item[2] > 0, self.data))], s=3, c="red", cmap=self.cmap,alpha=0.5)
            plt.scatter(outline[0], outline[1],s=1, color = "red", marker='_')
            fig.gca().fill(mask[0], mask[1], "red", alpha=0.5)
            # plt.plot(outline[0], outline[1],color = "red", marker='')
            save_margin(data=self.mask["margin"], save_path=self.image_path.replace(".png", "_margin.json"))

        # grid
        # plt.grid(linestyle="dotted", linewidth=1.5, alpha=0.5)
        plt.savefig(os.path.join(self.image_path))
        # plt.show()
        plt.close()



if __name__ == '__main__':
    def run(args):
        t1 = time.time()
        data_file = 'PinToPinSim.printSte0\\data_map_50'
        save_file = 'test_hspice\\hspice_eye.png'
        csv_file = 'test\\data.csv'
        # EyeDiagramPainter(data_file, save_file).paint_scatter()
        data_file = 'other\\data_map_50'
        # EyeDiagramPainter(data_file, save_file,data_rate="8533").paint_scatter()
        EyeDiagramPainter(csv_file, save_file, source="AMI",data_rate="6400").paint_scatter()
        t2 = time.time()
        print(f"Total time: {t2 - t1} s")


    run(sys.argv[1:])

# if __name__ == '__main__':
#     def run(args):

#         try:
#             # printSte0_file, save_folder = args[0], args[1]
#             printSte0_file, save_folder = "PinToPinSim.printSte0", "hspice_output"
#             printSte0_file, save_folder = "other", "hspice_output"
#             os.makedirs(save_folder, exist_ok=True)
#             filtered_files = []
#             pattern = re.compile(r'^data_map_\d+$')
#             for root, dirs, files in os.walk(printSte0_file):
#                 for file in files:
#                     if pattern.match(file):
#                         filtered_files.append(os.path.join(root, file))
#             if filtered_files:
#                 for data_file in filtered_files:
#                     save_file_name = os.path.basename(data_file).replace("data_map_", "port") + "_eye.png"
#                     save_file = os.path.join(save_folder, save_file_name)
#                     try:
#                         # EyeDiagramPainter(data_file, save_file).paint_scatter()
#                         EyeDiagramPainter(data_file, save_file,data_rate="8533").paint_scatter()
#                         print(f"\t\t{save_file_name}")
#                     except Exception as e:
#                         print(f"\t[Warning] Eye diagram generation failed! {save_file_name}")
#                         print(f"\t[Warning] {save_file_name} Eye diagram generation failed!")
#                         print(f"\t          The input file is: {data_file}")
#                         print(f"\t          Error details: {str(e)}")
#             else:
#                 print("The data_map_{num} file was not found in the PinToPinSim.printSte0 directory.")
#         except IndexError:
#             print("Args: 1.Data file path; 2.Save file path")


#     run(sys.argv[1:])
