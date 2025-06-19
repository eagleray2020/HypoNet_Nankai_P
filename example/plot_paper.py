import numpy as np
import matplotlib.pyplot as plt
import pygmt
from geopy.distance import geodesic

def project_point_onto_line_segment(A, B, P):
    """
    Project point P onto the line segment AB.

    Parameters:
    A, B: np.array, shape (2,)
        Endpoints of the line segment AB.
    P: np.array, shape (2,)
        Point to be projected.

    Returns:
    np.array, shape (2,)
        The projection of point P onto the line segment AB.
    """
    AP = P - A
    AB = B - A
    AB_norm_squared = np.dot(AB, AB)
    if AB_norm_squared == 0:
        return A  # A and B are the same point, return A

    t = np.dot(AP, AB) / AB_norm_squared
    t = np.clip(t, 0, 1)  # Ensure t is within [0, 1] to stay on the line segment

    return A + t * AB

def distances_from_A_to_projections(A, B, points):
    """
    Project each point in 'points' onto the line segment AB and calculate distances from A to the projections.

    Parameters:
    A, B: np.array, shape (2,)
        Endpoints of the line segment AB.
    points: np.array, shape (n, 2)
        Array of points to be projected.

    Returns:
    np.array, shape (n,)
        Distances from A to the projections of the points onto the line segment AB.
    """
    projections = np.array([project_point_onto_line_segment(A, B, P) for P in points])
    distances = np.linalg.norm(projections - A, axis=1)
    return distances



region = [134, 137.4, 32.3, 33.9]
line_lon = 135.75
line_s = np.array([line_lon, region[2]+0.1])
line_e = np.array([line_lon, region[3]-0.1])
dist_cart = geodesic(line_s[::-1],line_e[::-1]).kilometers
dist_lonlat = np.linalg.norm(line_s-line_e)

import argparse

parser = argparse.ArgumentParser(description="Plot paper script")
parser.add_argument('--x_s', type=int, default=None, help='Start index')
parser.add_argument('--x_e', type=int, default=None, help='End index')
args_cli = parser.parse_args()

x_s = args_cli.x_s
x_e = args_cli.x_e
ns = x_e - x_s + 1



xs_pinn_mean = np.zeros((ns, 3))
xs_pinn_std = np.zeros((ns, 3))
for i in range(ns):
    buf = np.genfromtxt("output/result."+str(i).zfill(8)+".txt")
    xs_pinn_mean[i,:] = buf[0,0:3]
    xs_pinn_mean[i,2] *= -1
    xs_pinn_std[i,:] = buf[0,3:6]

pygmt.config(FONT_ANNOT_PRIMARY="12p")
pygmt.config(FONT_ANNOT_SECONDARY="12p")


# lonlat map
fig = pygmt.Figure()

fig.basemap(region=region, projection="M12", frame=["agf", "WSne"])
fig.coast(shorelines="0.5", G="gray", S="white", map_scale="x10.5/0.8+c0+w50k+f+l")
tmpdata = np.vstack((xs_pinn_mean[:,0:2].T, (xs_pinn_std[:,0:2]*2).T)).T
fig.plot(data=tmpdata, error_bar="xyi0+w0.1p+p1p,magenta", style="c0.08c", fill="magenta", pen="magenta", label="HypoNet Nankai",)
fig.legend(position="JTL+jTL+o0.2c", box="+gwhite+p1p,black")
fig.plot(data=np.array([line_s, line_e]), fill="black", pen="1p,black",)

fig.savefig("hypo_lonlat.pdf")

# cross section map
coef = dist_cart/dist_lonlat

dist_mean = distances_from_A_to_projections(line_s, line_e, xs_pinn_mean[:,0:2])
dist_mean *= coef

xmin = 0.
xmax = 160.
ymin = -50.
ymax = 3.

fig = plt.figure(figsize=(5, 2.5))
hypo_size=3.
plt.errorbar(dist_mean, xs_pinn_mean[:,2], yerr=xs_pinn_std[:,2]*2, xerr=xs_pinn_std[:,1]*coef*2, elinewidth=1., markersize=hypo_size*0.5, fmt="o", label="PINN", color="m")
plt.xlabel("Distance (km)")
plt.ylabel("z (km)")
plt.xlim(0,160)
plt.ylim(-50, 3)
plt.gca().set_aspect(1)

plt.savefig("hypo_cross.pdf", bbox_inches="tight")