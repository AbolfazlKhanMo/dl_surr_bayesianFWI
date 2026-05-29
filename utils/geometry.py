import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from utils import get_z_interp

def check_topology_top(node1_x, node2_x):

    node_I_x  = node1_x
    node_I_z  = get_z_interp.oriskany_top(node_I_x)
    node_II_x = node2_x
    node_II_z = get_z_interp.oriskany_bottom(node_II_x)

    x_constraint = np.array([6500, 6208, 5716, 4916, 4527, 4226, 3978, 3689, 3438, 3129, 2867, 2674, 2495, 2324, 1993, 1780, 1457, 1215, 930, 640, 406, 226, 0])
    z_constraint = np.array([1648, 1655, 1661, 1662, 1685, 1738, 1824, 2012, 2231, 2446, 2553, 2576, 2546, 2496, 2367, 2283, 2197, 2176, 2202, 2250, 2309, 2334, 2336])
    # Sort the constraint points based on x values
    sorted_indices = np.argsort(x_constraint)
    x_sorted = x_constraint[sorted_indices]
    z_sorted = z_constraint[sorted_indices]

    # Create a B-spline representation
    tck = splrep(x_sorted, z_sorted)

    # Generate new x values for interpolation
    x_new = np.linspace(min(x_sorted), max(x_sorted), 6500)
    y_new = splev(x_new, tck)

    # Function to check if the line intersects the constraint curve
    def intersects_line_with_curve(node_I_x, node_I_z, node_II_x, node_II_z, x_new, y_new, tck):
        x_values = np.linspace(min(node_I_x, node_II_x), max(node_I_x, node_II_x), 6500)
        line_z_values = node_I_z + (node_II_z - node_I_z) * (x_values - node_I_x) / (node_II_x - node_I_x)

        for i in range(len(x_new) - 1):
            if x_new[i] > x_values[0] and x_new[i] < x_values[-1]:
                curve_y_value = splev(x_new[i], tck)
                line_y_value = node_I_z + (node_II_z - node_I_z) * (x_new[i] - node_I_x) / (node_II_x - node_I_x)
                if (line_y_value > curve_y_value and line_z_values[i + 1] < curve_y_value) or \
                   (line_y_value < curve_y_value and line_z_values[i + 1] > curve_y_value):
                    return True
        return False

    # Check for intersection and return the appropriate statement
    if intersects_line_with_curve(node_I_x, node_I_z, node_II_x, node_II_z, x_new, y_new, tck):
        return "Singular Topology!"
    else:
        return "Topology is OK"

def check_topology_bottom(node1_x, node2_x):

    node_I_x  = node1_x
    node_I_z  = get_z_interp.oriskany_top(node_I_x)
    node_II_x = node2_x
    node_II_z = get_z_interp.oriskany_bottom(node_II_x)

    x_constraint = np.array([6500, 5945, 5458, 4721, 4201, 3763, 3416, 3116, 2877, 2560, 2294, 1990, 1601, 1037, 528, 0])
    z_constraint = np.array([1164, 1202, 1240, 1285, 1282, 1311, 1425, 1611, 1779, 1924, 1894, 1762, 1660, 1630, 1651, 1726])
    # Sort the constraint points based on x values
    sorted_indices = np.argsort(x_constraint)
    x_sorted = x_constraint[sorted_indices]
    z_sorted = z_constraint[sorted_indices]

    # Create a B-spline representation
    tck = splrep(x_sorted, z_sorted)

    # Generate new x values for interpolation
    x_new = np.linspace(min(x_sorted), max(x_sorted), 6500)
    y_new = splev(x_new, tck)

    # Function to check if the line intersects the constraint curve
    def intersects_line_with_curve(node_I_x, node_I_z, node_II_x, node_II_z, x_new, y_new, tck):
        x_values = np.linspace(min(node_I_x, node_II_x), max(node_I_x, node_II_x), 6500)
        line_z_values = node_I_z + (node_II_z - node_I_z) * (x_values - node_I_x) / (node_II_x - node_I_x)

        for i in range(len(x_new) - 1):
            if x_new[i] > x_values[0] and x_new[i] < x_values[-1]:
                curve_y_value = splev(x_new[i], tck)
                line_y_value = node_I_z + (node_II_z - node_I_z) * (x_new[i] - node_I_x) / (node_II_x - node_I_x)
                if (line_y_value > curve_y_value and line_z_values[i + 1] < curve_y_value) or \
                   (line_y_value < curve_y_value and line_z_values[i + 1] > curve_y_value):
                    return True
        return False

    # Check for intersection and return the appropriate statement
    if intersects_line_with_curve(node_I_x, node_I_z, node_II_x, node_II_z, x_new, y_new, tck):
        return "Singular Topology!"
    else:
        return "Topology is OK"

def plot_topology(name, path):

    #                                 ########### Given data points for the constraint curve ###########
    # Horizon I - Rome Formation ‌Bottom
    x_i   = np.array([6500, 6131, 5271, 4764, 4348, 3836, 3304, 2906, 2424, 2402, 2190, 1912, 1719, 1576])
    z_i   = np.array([1772, 1814, 1920, 2010, 2104, 2253, 2472, 2670, 2995, 3011, 3279, 3610, 3891, 4300])

    # Horizon II - Mississipian Age Formation ‌Bottom
    x_ii  = np.array([1576, 1538, 1484, 1376, 1224, 1067, 937, 782, 609, 473])
    z_ii  = np.array([4300, 4090, 3999, 3931, 3936, 4011, 4093, 4165, 4234, 4300])

    # Horizon III - Forkenobs & Brallier ‌Bottom
    x_iii = np.array([2402, 2229, 2009, 1709, 1432, 975.3, 654, 331.5, 0])
    z_iii = np.array([3011, 2808, 2568, 2383, 2367, 2526, 2702, 2902, 3132])

    # Horizon IV - Pulsaki Thrust Sheet ‌Bottom
    x_iv   = np.array([6500, 6262, 5482, 4907, 4442, 4061, 3760, 3574, 3400, 3181, 2982, 2737, 2486, 2222, 1998, 1703, 1446, 1176, 910, 679, 363, 153.8, 0])
    z_iv   = np.array([1658, 1664, 1680, 1685, 1719, 1808, 1981, 2136, 2280, 2431, 2527, 2584, 2546, 2469, 2381, 2272, 2200, 2169, 2218, 2257, 2333, 2325, 2338])

    # Horizon V - Oriskany Sandstone ‌Bottom
    x_v    = np.array([6500, 5945, 5458, 4721, 4201, 3763, 3416, 3116, 2877, 2560, 2294, 1990, 1601, 1037, 528, 0])
    z_v    = np.array([1164, 1202, 1240, 1285, 1282, 1311, 1425, 1611, 1779, 1924, 1894, 1762, 1660, 1630, 1651, 1726])

    # Horizon VI - Martinsburg Formation ‌Bottom
    x_vi   = np.array([6500, 6284, 5975, 5355, 4852, 4446, 3940, 3576, 3218, 3015, 2855, 2616, 2398, 2145, 1945, 1606, 1178, 742, 307, 0])
    z_vi   = np.array([535, 562.5, 596, 652, 674.8, 693.6, 728.4, 831.6, 1009, 1174, 1316, 1437, 1355, 1182, 1059, 959, 975.5, 1057, 1135, 1173])


    # List of constraint curves
    constraints = [
        (x_i, z_i, 'Rome Formation'),
        (x_ii, z_ii, 'Mississippian Age Formation'),
        (x_iii, z_iii, 'Fokenobs & Brallier Formation'),
        (x_iv, z_iv, 'Pulsaki Thrust Sheet'),
        (x_v, z_v, 'Oriskany Sandstone'),
        (x_vi, z_vi, 'Martinsburg Formation')
    ]

    # Plot the constraint curves
    plt.figure(figsize=(10, 6))
    # Define the rectangle width and height
    rect_width = 6500
    rect_height = 4300
    # Plot the rectangle
    plt.plot([0, rect_width, rect_width, 0, 0], [0, 0, rect_height, rect_height, 0], 'k-')

    for x, z, label in constraints:
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        z_sorted = z[sorted_indices]

        # Create a B-spline representation
        tck = splrep(x_sorted, z_sorted)

        # Generate new x values for interpolation
        x_new = np.linspace(min(x_sorted), max(x_sorted), 300)
        z_new = splev(x_new, tck)

        # Plot the constraint curve
        plt.plot(x_new, z_new, label=label, linestyle='--')

    # creating domain
    ticks = np.linspace(0, 4300, 11)
    labels = ["4300", "3870", "3440", "3010", "2580", "2150", "1720", "1290", "860", "430", "0"]

    # plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, label='Proposed Line')

    # Plot utils
    plt.xlabel('X-position [m]')
    plt.ylabel('Depth [m]')
    plt.yticks(ticks=ticks, labels=labels)
    plt.xlim(0, rect_width)
    plt.ylim(0, rect_height)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, name), format='png', dpi=300)
    plt.close()

# Example usage:
# node_I_x, node_I_z = 6011, 1197  # First point
# node_II_x, node_II_z = 3366, 2301  # Second point
# result = check_topology(node_I_x, node_I_z, node_II_x, node_II_z)
# print(result)