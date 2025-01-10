
import os
import numpy as np
import bisect
import json



def max_gap(datas,y_axis = False):
            sorted_data = [point[0] for point in datas]
            if y_axis:
                sorted_data = [point[1] for point in datas]

            max_gap = 0
            pair = (None, None)

            for i in range(len(sorted_data) - 1):
                gap = sorted_data[i + 1] - sorted_data[i]
                if gap > max_gap:
                    max_gap = gap
                    pair = (datas[i], datas[i + 1])
            
            if max_gap == 0:
                return None
            
            return (max_gap, pair)

def normalize_c(value_table):
    table = np.where(value_table > 0, value_table, 10000)
    value_min, value_max = min(table), max(value_table)

    value_table = np.where(
        value_table > 0,
        (value_table - value_min) * 0.9 / (value_max - value_min) + 0.00002,
        # (value_table - value_min) / (value_max - value_min),
        value_table,
    )
    return value_table



def calculate_top_and_bottom(pair,mask_bottom,mask_top):
    bottom,top = mask_bottom - pair[0][1],pair[1][1] - mask_top
    return top,bottom


def calculate_top_and_bottom_by_judge_value(judge_value,x_pairs,mask_bottom,mask_top,margin = False):

    x_list = [item[0][0] for item in x_pairs]

    if judge_value in x_list:
        y_pair = x_pairs[x_list.index(judge_value)]
        top,bottom = calculate_top_and_bottom(y_pair,mask_bottom,mask_top)

    else:
        pair_index1,pair_index2 = bisect.bisect_left(x_list, judge_value),bisect.bisect_right(x_list, judge_value)
        try:
            y_pair1 = x_pairs[pair_index1]
            top1,bottom1 = calculate_top_and_bottom(y_pair1,mask_bottom,mask_top)
        except Exception:
            top1,bottom1 = 0,0
        
        try:
            y_pair2 = x_pairs[pair_index2]
            top2,bottom2 = calculate_top_and_bottom(y_pair2,mask_bottom,mask_top)
        except Exception:
            top2,bottom2 = 0,0
        
        top,bottom = (top1 + top2)/2,(bottom1 + bottom2)/2

    if margin:
        top = None if top <= 0 else top
        bottom = None if bottom <= 0 else bottom
    
    return top,bottom

def calculate_set_up_and_hold(pair,mask_left,mask_right):
    setup,hold = mask_left - pair[0][0],pair[1][0] - mask_right
    return setup,hold


def calculate_set_up_and_hold_by_judge_value(judge_value,y_pairs,mask_left,mask_right,margin = False):

    y_list = [item[0][1] for item in y_pairs]

    if judge_value in y_list:
        x_pair = y_pairs[y_list.index(judge_value)]
        setup,hold = calculate_set_up_and_hold(x_pair,mask_left,mask_right)

    else:
        pair_index1,pair_index2 = bisect.bisect_left(y_list, judge_value),bisect.bisect_right(y_list, judge_value)
        
        try:
            x_pair1 = y_pairs[pair_index1]
            setup1,hold1 = calculate_set_up_and_hold(x_pair1,mask_left,mask_right)
        except Exception:
            setup1,hold1 = 0,0

        try:
            x_pair2 = y_pairs[pair_index2]
            setup2,hold2 = calculate_set_up_and_hold(x_pair2,mask_left,mask_right)
        except Exception:
            setup2,hold2 = 0,0

        setup,hold = (setup1 + setup2)/2,(hold1 + hold2)/2

    if margin:
        setup = None if setup <= 0 else setup
        hold = None if hold <= 0 else hold
    
    return setup,hold

def splicing_line_segments(line1,line2 = []):
        if not line1:
            return line2
        if not line2:
            return line1
        
        common_elements = [item for item in line1 if item in line2]
        if common_elements:
            line2 = [item for item in line2 if item not in line1]
            if not line2:
                return line1

        l1_start = line1[0]
        l1_end = line1[-1]
        l2_start = line2[0]
        l2_end = line2[-1]

        d1 = calcute_distance(l1_start,l2_start)
        d2 = calcute_distance(l1_start,l2_end)
        d3 = calcute_distance(l1_end,l2_start)
        d4 = calcute_distance(l1_end,l2_end)

        d_judge = min(d1,d2,d3,d4)
        if d_judge == d1:
            return line1[::-1] + line2
        if d_judge == d2:
            return line2 + line1
        if d_judge == d3:
            return line1 + line2
        if d_judge == d4:
            return line1 + line2[::-1]

def calcute_distance(p1,p2):
    x1,y1 = p1[0],p1[1]
    x2,y2 = p2[0],p2[1]
    return (x1-x2)**2 + (y1-y2)**2

def remove_duplicates(lst):
    seen = set() 
    result = [] 
    for item in lst:
        if item not in seen:
            result.append(item)  
            seen.add(item) 
    return result

def calculate_offest_x(x_middle,y_middle,gap_pair_x,tdivw1,tdivw2,vdivw):
    y_judge_key = [y_middle + vdivw/2,
                    y_middle,
                    y_middle - vdivw/2,]
    y_pairs = sorted([item[1] for item in gap_pair_x],key=lambda item: item[0][1])
    
    setup_hold = []

    for index_judge_value,judge_value in enumerate(y_judge_key):
        if index_judge_value % 2:
            setup,hold = calculate_set_up_and_hold_by_judge_value(judge_value,y_pairs,(x_middle- tdivw1*1e12/2),(x_middle + tdivw1*1e12/2))
        else:    
            setup,hold = calculate_set_up_and_hold_by_judge_value(judge_value,y_pairs,(x_middle- tdivw2*1e12/2),(x_middle + tdivw2*1e12/2))
        setup_hold.append((setup,hold))
    
    offset_x = 0
    for values in [setup_hold[0]]:
        offset_x += (values[1] - values[0])/2
    
    return offset_x

def calculate_offest_y(x_middle,y_middle,gap_pair_y,tdivw1,tdivw2,vdivw):
    x_judge_key = [x_middle - tdivw1*1e12/2,
                   x_middle - tdivw2*1e12/2,
                   x_middle + tdivw2*1e12/2,
                   x_middle + tdivw1*1e12/2,]
    x_pairs = sorted([item[1] for item in gap_pair_y],key=lambda item: item[0][0])
    
    top_bottom = []

    for index_judge_value,judge_value in enumerate(x_judge_key):
        if index_judge_value == 0 or index_judge_value == 3:
            top,bottom = calculate_top_and_bottom_by_judge_value(judge_value,x_pairs,y_middle,y_middle)
        else:    
            top,bottom = calculate_top_and_bottom_by_judge_value(judge_value,x_pairs,y_middle - vdivw/2,y_middle + vdivw/2)
        top_bottom.append((top,bottom))
    
    offset_y = 0
    for values in [top_bottom[3]]:
        offset_y += (values[0] - values[1])/2
    
    return offset_y

def calculate_center(x_middle,y_middle,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw):
    offset_x = calculate_offest_x(x_middle,y_middle,gap_pair_x,tdivw1,tdivw2,vdivw)
    offset_y = calculate_offest_y(x_middle,y_middle,gap_pair_y,tdivw1,tdivw2,vdivw)
    if abs(offset_x) <= 5e-3 and abs(offset_y) <= 5e-3:
        return x_middle,y_middle
    else:
        return calculate_center(x_middle+offset_x,y_middle+offset_y,gap_pair_x,gap_pair_y,tdivw1,tdivw2,vdivw)


def save_margin(data,save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, "w") as json_file:
        json.dump(data, json_file)









