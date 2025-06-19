import typing
from HypoNet_Nankai.HypoNet.ellipsoid import Ellipsoid
import pymap3d
import torch 

torch.set_default_dtype(torch.float64)

try:
   # from numpy import hypot, cos, sin, arctan2 as atan2, radians, pi, asarray
    from torch import hypot, cos, sin, arctan2 as atan2, deg2rad as radians, asarray, sqrt
    from numpy import pi
except ImportError:
    from math import atan2, hypot, cos, sin, radians, pi

    asarray = None

#__all__ = ["cart2pol", "pol2cart", "cart2sph", "sph2cart", "sign"]
if typing.TYPE_CHECKING:
    from numpy import ndarray

def geodetic2ecef(lat, lon, alt, ell: Ellipsoid = None, deg: bool = True):
    """
    point transformation from Geodetic of specified ellipsoid (default WGS-84) to ECEF

    Parameters
    ----------

    lat : float
           target geodetic latitude
    lon : float
           target geodetic longitude
    h : float
         target altitude above geodetic ellipsoid (meters)
    ell : Ellipsoid, optional
          reference ellipsoid
    deg : bool, optional
          degrees input/output  (False: radians in/out)


    Returns
    -------

    ECEF (Earth centered, Earth fixed)  x,y,z

    x : float
        target x ECEF coordinate (meters)
    y : float
        target y ECEF coordinate (meters)
    z : float
        target z ECEF coordinate (meters)
    """
    if ell is None:
        ell = Ellipsoid()

#    lat, ell = sanitize(lat, ell, deg)
    if deg:
        lat = radians(lat)
        lon = radians(lon)

    # radius of curvature of the prime vertical section
    N = ell.semimajor_axis ** 2 / sqrt(ell.semimajor_axis ** 2 * cos(lat) ** 2 + ell.semiminor_axis ** 2 * sin(lat) ** 2)
    # Compute cartesian (geocentric) coordinates given  (curvilinear) geodetic
    # coordinates.
    x = (N + alt) * cos(lat) * cos(lon)
    y = (N + alt) * cos(lat) * sin(lon)
    z = (N * (ell.semiminor_axis / ell.semimajor_axis) ** 2 + alt) * sin(lat)

    return x, y, z

def geodetic2enu(lat, lon, h, lat0, lon0, h0, ell: Ellipsoid = None, deg: bool = True):
    """
    Parameters
    ----------
    lat : float
          target geodetic latitude
    lon : float
          target geodetic longitude
    h : float
          target altitude above ellipsoid  (meters)
    lat0 : float
           Observer geodetic latitude
    lon0 : float
           Observer geodetic longitude
    h0 : float
         observer altitude above geodetic ellipsoid (meters)
    ell : Ellipsoid, optional
          reference ellipsoid
    deg : bool, optional
          degrees input/output  (False: radians in/out)


    Results
    -------
    e : float
        East ENU
    n : float
        North ENU
    u : float
        Up ENU
    """
    x1, y1, z1 = geodetic2ecef(lat, lon, h, ell, deg=deg)
#    x2, y2, z2 = pymap3d.geodetic2ecef(lat0, lon0, h0, ell, deg=deg)
    x2, y2, z2 = pymap3d.geodetic2ecef(lat0, lon0, h0, deg=deg)

    return pymap3d.uvw2enu(x1 - x2, y1 - y2, z1 - z2, lat0, lon0, deg=deg)
