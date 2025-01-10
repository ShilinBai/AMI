import numpy as np



def find_zone(value_table):

    table = np.where(value_table > 0, 1, 0)
    # table = np.transpose(table)
    pairs = find_pairs(table)
    all_groups = find_groups(pairs)
    largest_region = find_eye_max(all_groups)
    # v_level = value_to_pixel_length(
    #                     config.aperture_h,
    #                     config.grid_y.value + 200,
    #                     config.v_max.value,
    #                     config.v_min.value,
    #                 )
    # meas = get_aperture(largest_region, int(v_level))
    return largest_region


def find_pairs(image):
        x_size, y_size = image.shape
        pairs = []

        for x in range(x_size):
            current_pair = []
            y1 = 0
            y2 = 0
            for y in range(1, y_size):
                if image[x, y] == 0 and image[x, y - 1] == 1:
                    y1 = y

                if y1 > 0 and (image[x, y] == 1 and image[x, y - 1] == 0):
                    y2 = y
                if y2 > 0 and ((y2 - y1) > 1):
                    current_pair.append([(x, y1), (x, y2)])
                    y1 = 0
                    y2 = 0
            pairs.append(current_pair)

        return pairs

def find_groups(coord_lists):
        all_groups = []
        last_group_list = []
        for current_coord_list in coord_lists:
            last_group_list.append([[[0, 1e10], [0, 1e10]]])
            current_group_list = []

            ind_last = 0
            ind_current = 0
            while ind_last < len(last_group_list) and ind_current < len(
                current_coord_list
            ):
                current_coord = current_coord_list[ind_current]
                last_group = last_group_list[ind_last]
                last_coord = last_group[-1]

                if current_coord[0][1] >= last_coord[1][1]:
                    # current coordinate pair is higher than the previous pair, move the previous pair
                    ind_last += 1
                elif last_coord[0][1] >= current_coord[1][1]:
                    # previous pair is higher than the current pair, create a new group
                    new_group = [current_coord]
                    all_groups.append(new_group)
                    current_group_list.append(new_group)
                    ind_current += 1
                else:
                    # current pair intersects with previous pair, add to the existing group
                    last_group.append(current_coord)
                    current_group_list.append(last_group)
                    ind_last += 1
                    ind_current += 1

            last_group_list = current_group_list

        return all_groups

def find_eye_max(all_groups):
        eye_max = None
        diff = []
        for group in all_groups:
            diff.append(group[-1][0][0] - group[0][0][0])
        ind = np.argmax(diff)
        eye_max = all_groups[ind]

        return eye_max

def value_to_pixel_length(value, coord, max_y, min_y):
        y_proportion = value / (max_y - min_y)
        y_pixel_length = (coord + 1) * y_proportion
        return y_pixel_length

def get_aperture(largest_region, x_level):
        x_list = np.array([coord[0][0] for coord in largest_region])
        y_list1 = np.array([coord[0][1] for coord in largest_region])
        y_list2 = np.array([coord[1][1] for coord in largest_region])
        box = []
        diff = []
        for i in range(len(x_list) - 1):
            x1 = x_list[i]
            for j in range(i + 1, len(x_list)):
                x2 = x_list[j]
                if x2 - x1 == x_level:
                    box.append([i, j])
        if box:
            t_high = None
            for ind_box in box:
                i = ind_box[0]
                j = ind_box[1]
                if y_list2[i] <= y_list2[j] and y_list1[i] >= y_list1[j]:
                    t1 = y_list2[i]
                    t2 = y_list1[i]
                    diff.append(t1 - t2)
                    ind = np.argmax(diff)
                    ind_i = box[ind][0]
                    ind_j = box[ind][1]
                    t_high = y_list2[ind_i]
                    t_low = y_list1[ind_i]
            if not t_high:
                return None
        else:
            return None
        return [t_low, t_high, x_list[ind_i], x_list[ind_j], diff[ind]]



