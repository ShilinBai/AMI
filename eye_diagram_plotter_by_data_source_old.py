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
    def __init__(self, path, save, source='HSPICE', width=10, height=6, cmap='HSPICE', background='black',
                 center=False):
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
        # self.data = self._get_data(path, center)
        self.data = self._get_data_csv(path)

        self.cmap = cmap
        self.width = width
        self.height = height
        self.background = background
    
    def _get_data_csv(self, path):
        df = pd.read_csv(path)
        x_list = np.array(df['__UnitInterval []'])
        y_list = np.array(df['__Amplitude [V]']) * 1e3
        c_list = np.array(df['EyeAfterProbe<b_input_22.int_ami_rx> []'])
        data = []
        for ind_x,x in enumerate(x_list):
            data.append((x,y_list[ind_x],c_list[ind_x]))
        return data

    def _get_data(self, path, center):
        # return:  data: {list {tuple}}: [(x0,y0,color),(x1,y1,color)...]
        if self.source == "HSPICE":
            return DataSourceHandler.handle_hspice_eye_diagram_data(path, center)
        elif self.source == "ADS":
            return DataSourceHandler.handle_ads_eye_diagram_data(path, center)
        else:
            raise AttributeError

    def _get_scatter_points(self):
        x, y, c = [], [], []
        for point in self.data:
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
        # self._get_scatter_points()
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        # paint scatter
        x, y, c = self._get_scatter_points()
        # set color bar ande background
        self.set_cmap_and_background(ax)
        pc = ax.scatter(x, y, s=50, c=c, cmap=self.cmap)
        # paint color bar
        # cb = fig.colorbar(pc, ax=ax)

        # def format_func(value, tick_number):
        #     return f"{round(value, 1)}m"

        # cb.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        # cb.set_label("Density", loc="top")
        # axis
        plt.gca().set_xlim([min(x), max(x)])
        plt.gca().set_ylim([min(y), max(y)])
        plt.xlabel("UnitInterval")
        plt.ylabel("Amplitude [mV]")
        fig.gca().fill([min(x),0.5 * (min(x)+ max(x)), max(x)], [min(y), max(y),min(y)], "blue")
        # grid
        # plt.grid(linestyle="dotted", linewidth=1.5, alpha=0.5)
        plt.savefig(os.path.join(self.image_path))
        # plt.show()
        plt.close()


if __name__ == '__main__':
    def run(args):
        t1 = time.time()

        try:
            data_file = 'test\\data_map_1'
            save_file = 'test\\csv_eye.png'
            csv_file = 'test\\data.csv'
            try:
                # EyeDiagramPainter(data_file, save_file).paint_scatter()
                EyeDiagramPainter(csv_file, save_file).paint_scatter()
                print(f"\t\t{save_file}")
            except Exception as e:
                print(f"\t          Error details: {str(e)}")
        except IndexError:
            print("Args: 1.Data file path; 2.Save file path")
        
        t2 = time.time()
        print(f"Total time: {t2 - t1} s")


    run(sys.argv[1:])
