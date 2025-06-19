import torch
import math


def initialize_hypocenter_location_TT(x_obs_p, y_obs_p, xmin_data=130., xmax_data=141., ymin_data=29., ymax_data=37.):
    # xmin_data, xmax_data, ymin_data, ymax_data: domain of the data given from GEBCO
    """
    Initialize the hypocenter location parameters for optimization
    and determine the lower and upper bounds of horizontal domain for optimization.

    The horizontal domain size is determined by 2*(minimum travel time)*(maximum velocity).
    the center of the domain is set to be the coordinate with minimum travel time.
    The vertical domain is set to be -59. to -2. km.

    
    Args:
        x_obs_p: Observation points for P-waves
        y_obs_p: Observation data for P-waves
        xmin_data, xmax_data, ymin_data, ymax_data: domain of the data given from GEBCO

        
    Returns:
        tuple: (xs, a, b) where:
            - xs: Initial hypocenter location
            - a: Lower bounds for optimization
            - b: Upper bounds for optimization
    """
    def deg_per_km_approx(lat_deg):
        lat_rad = math.radians(lat_deg)
        # Earth's radius in kilometers
        R = 6371.0
        # Latitude direction: 1 degree ≈ 111.32 km (nearly constant)
        deg_per_km_lat = 1 / 111.32
        # Longitude direction: depends on latitude (cosine factor)
        deg_per_km_lon = 1 / (111.32 * math.cos(lat_rad))
        return deg_per_km_lat, deg_per_km_lon
    
    vp_max = 8.
    ind = torch.argmin(y_obs_p[:,0])
    y_obs_min = y_obs_p[ind,0]
    max_dist_km = y_obs_min*vp_max # maximum distance of hypocenter from the observation point
    lat = x_obs_p[ind,1]
    deg_per_km_lat, deg_per_km_lon = deg_per_km_approx(lat)
    #print("max_dist_km, deg_per_km_lat, deg_per_km_lon:", max_dist_km, deg_per_km_lat, deg_per_km_lon)
    max_dist_lat = max_dist_km*deg_per_km_lat
    max_dist_lon = max_dist_km*deg_per_km_lon
    xmin_loc = max(x_obs_p[ind,0]-0.5, xmin_data)
    xmax_loc = min(x_obs_p[ind,0]+0.5, xmax_data)
    ymin_loc = max(x_obs_p[ind,1]-0.5, ymin_data)
    ymax_loc = min(x_obs_p[ind,1]+0.5, ymax_data)
    zmin_loc = -59.
    zmax_loc = -2.
    
    a = torch.tensor([xmin_loc, ymin_loc, zmin_loc])
    b = torch.tensor([xmax_loc, ymax_loc, zmax_loc])
    
    # Setting initial value of hypocenter location
    xs = torch.rand((3))
    xs[0] = (xmax_loc-xmin_loc)*0.5 + xmin_loc
    xs[1] = (ymax_loc-ymin_loc)*0.5 + ymin_loc
    xs[2] = -30.
    
    return xs, a, b