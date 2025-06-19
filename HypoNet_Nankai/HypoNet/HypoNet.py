import torch
import HypoNet_Nankai.HypoNet.pymap3d_torch as pymap3d_torch
import numpy as np
#import mypinn_FFswish_gpu_3D_epsilon as mypinn
import netCDF4
import HypoNet_Nankai.HypoNet.geodetic as geodetic
import HypoNet_Nankai.HypoNet.mymultipinn as mymultipinn
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#from math import cos, sin
#from numba import jit
def rotate_translate_matmul(coor, origin, rot):
    """Apply rotation and translation to coordinates."""
    coor_mod = torch.empty_like(coor)
    coor_mod[:,2] = coor[:,2] 
    coor_mod[:,0:2] = torch.subtract(coor[:,0:2],origin)
    coor_mod[:,0:2] = (rot@coor_mod[:,0:2].T).T
    return coor_mod

class PINNtomo_param:
    def read(self):
#        if im == 0:
        
        with open(BASE_DIR+"/input/setting") as f:
            s = f.readline()
            s = f.readline()
            items = s.split()
            self.nhlt = int(items[0])
            self.nnt = int(items[1])
            self.fact_name = items[2]
#            self.nhlv = int(items[2])
#            self.nnv = int(items[3])
            self.nmff = len(items)-3
            self.FFsigma = []
            for i in range(self.nmff):
                self.FFsigma.append(float(items[i+3]))
            """
            s = f.readline()
            s = f.readline()
            items = s.split()
            self.nhlt = int(items[0])
            self.nnt = int(items[1])
#            self.nhlv = int(items[2])
#            self.nnv = int(items[3])
            self.FFsigma = float(items[2])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.max_iter = int(items[0])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.lr = float(items[0])
#            self.lrT = float(items[1])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.bs_eik = int(items[0])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.deviceid = int(items[0])
            """
class model():
    def __init__(self, inputdir, device=0):
        BFP =  PINNtomo_param()
        BFP.read()
        coor = np.genfromtxt(BASE_DIR+"/"+inputdir+"/coor_minmax.txt")
        xmin = coor[0,0]
        xmax = coor[0,1]
        ymin = coor[1,0]
        ymax = coor[1,1]
        zmin = coor[2,0]
        zmax = coor[2,1]
        vmin = coor[3,0]
        vmax = coor[3,1]

        self.cart_corners = torch.from_numpy(np.genfromtxt(BASE_DIR+"/input/IL1-750_cart_corners.txt"))
        rot_deg = -(90+torch.rad2deg(torch.arctan2(self.cart_corners[3,1]-self.cart_corners[4,1],self.cart_corners[3,0]-self.cart_corners[4,0])))
        x = torch.deg2rad(rot_deg)
        self.rot = torch.tensor([[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]])

        self.lon0 = 136.
        self.lat0 = 33.
        self.h0   = 0.

        self.domain_lonlat = torch.from_numpy(np.genfromtxt(BASE_DIR+"/input/modeldomain_lonlat_Agata2025SRL.txt"))
        self.domain_z = torch.tensor([-59000.0, 0.0]) # in m
#N        self.leveling  = 24.39 # Tokyo Peil in m
        #model_thres[:,2]=-model_thres[:,2]+leveling     



#        mypinn_t: torch.nn.Module = mypinn.MyPINN3D_t(BFP.nhlt, BFP.nnt)#.to(device)
#        mypinn_t.set_coor_minmax(xmin, xmax, ymin, ymax, zmin, zmax)
#        mypinn_t.set_v_minmax_from_scalar(vmin,vmax)
#       state_dict = torch.load("input/mypinn_t_state_dict.pt")#.to(device2)
#        mypinn_t.load_state_dict(state_dict)
            
        if device == 0: #cpu
            self.device = torch.device("cpu")
        elif device == 1:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")

        self.mypinn_t = mymultipinn.MyMultiPINN(inputdir, BFP.fact_name, BFP.nhlt, BFP.nnt, BFP.nmff, BFP.FFsigma)
        self.mypinn_t.input_models(inputdir, self.device)
#        self.mypinn_t = mypinn_t.to(self.device)

        gebco = netCDF4.Dataset(BASE_DIR+"/input/gebco_2023_n37.0_s29.0_w130.0_e141.0.nc")
        lat = gebco.variables["lat"][:]
        lon = gebco.variables["lon"][:]
        self.latmin_gebco = np.min(lat)
        self.lonmin_gebco = np.min(lon)
        self.dlonlat_gebco = lon[1]-lon[0]
        self.elev = torch.from_numpy(gebco.variables["elevation"][:])#.to(self.device)
        #geoid_path = BASE_DIR+"/input/EGM2008_Japan.npy"
        geoid_path = BASE_DIR+"/input/EGM2008_Nankai.npz"
        self.myegm = geodetic.EGM2008(geoid_path)
    
    def is_within_domain(self, lon, lat, h=None):
        """
        Check if the input coordinates (lon, lat, h) are within the model domain.

        Args:
            lon (torch.Tensor or np.ndarray): Longitudes, shape (N,)
            lat (torch.Tensor or np.ndarray): Latitudes, shape (N,)
            h (torch.Tensor or np.ndarray): Elevations (km), shape (N,)

        Returns:
            torch.BoolTensor: Boolean mask of shape (N,) indicating if each point is within the domain.
        """
        # Convert inputs to torch tensors if needed
        if not torch.is_tensor(lon):
            lon = torch.tensor(lon)
        if not torch.is_tensor(lat):
            lat = torch.tensor(lat)


        # Ensure all are 1D and same shape
        lon = lon.flatten()
        lat = lat.flatten()


        # Polygon: self.domain_lonlat (shape [5,2]), closed polygon
        poly = self.domain_lonlat
        x = lon
        y = lat

        # Ray casting algorithm for point-in-polygon
        nvert = poly.shape[0]
        inside = torch.zeros_like(x, dtype=torch.bool)
        for i in range(nvert - 1):
            xi, yi = poly[i]
            xj, yj = poly[i + 1]
            cond1 = ((yi > y) != (yj > y))
            slope = (xj - xi) / (yj - yi + 1e-12)
            xints = slope * (y - yi) + xi
            cond2 = x < xints
            inside ^= cond1 & cond2
        if h is None:
            return inside
        else:
            # Elevation check
            if not torch.is_tensor(h):
                h = torch.tensor(h)
            h = h.flatten()
            zmin = self.domain_z[0]
            zmax = self.domain_z[1]
            elev_mask = (h >= zmin) & (h <= zmax)

            mask = inside & elev_mask
            return mask




    def gebco_nearestneighbor(self, r_lon, r_lat):
        ind_lon = torch.round((r_lon-self.lonmin_gebco)/self.dlonlat_gebco).to(torch.int32)
        ind_lat = torch.round((r_lat-self.latmin_gebco)/self.dlonlat_gebco).to(torch.int32)
#        print(ind_lon, ind_lat)
        return self.elev[ind_lat, ind_lon]

    def inference_numpy(self, s_lon_np, s_lat_np, s_h_np, r_lon_np, r_lat_np):
        # input elevation is in km
        s_lon = torch.from_numpy(s_lon_np)
        s_lat = torch.from_numpy(s_lat_np)
        s_h   = torch.from_numpy(s_h_np)*1e+3 #km=>m
        s_lon.requires_grad = True
        s_lat.requires_grad = True
        s_h.requires_grad = True
        r_lon = torch.from_numpy(r_lon_np)
        r_lat = torch.from_numpy(r_lat_np)
        r_geo = torch.from_numpy(self.myegm.interpolate(r_lat_np, r_lon_np))
        s_geo = torch.from_numpy(self.myegm.interpolate(s_lat_np, s_lon_np))
        #x,y,z=geodetic2enu(random_point[:,1],random_point[:,0],self.elev[ilatlist,ilonlist]+geoid,self.lat0,self.lon0,self.h0) #km=>
        r_h   = self.gebco_nearestneighbor(r_lon, r_lat).clone().detach()#.to(torch.double)
        #s_h = s_h + self.leveling # adding Tokyo Peil height to elevation
        print("r_h(m)", r_h)

        if self.device != torch.device("cpu"):
            torch.set_default_tensor_type(cuda_ttype)
            s_lon = s_lon.to(self.device)
            s_lat = s_lat.to(self.device)
            s_h   = s_h.to(self.device)
            r_lon = r_lon.to(self.device)
            r_lat = r_lat.to(self.device)
            r_h   = r_h.to(self.device)

        s_e, s_n, s_u = pymap3d_torch.geodetic2enu(s_lat, s_lon, s_h+s_geo, self.lat0, self.lon0, self.h0)
        r_e, r_n, r_u = pymap3d_torch.geodetic2enu(r_lat, r_lon, r_h+r_geo, self.lat0, self.lon0, self.h0)
        s_enu = torch.vstack((torch.vstack((s_e, s_n)), s_u)).T*1e-3 #m=>km
        r_enu = torch.vstack((torch.vstack((r_e, r_n)), r_u)).T*1e-3 #m=>km
#        s_enu = rotate_translate(s_enu, self.cart_corners[3], self.rot)
#        r_enu = rotate_translate(r_enu, self.cart_corners[3], self.rot)
        s_enu = rotate_translate_matmul(s_enu, self.cart_corners[3], self.rot)
        r_enu = rotate_translate_matmul(r_enu, self.cart_corners[3], self.rot)
        print("r_enu(km)", r_enu)
        print("s_enu(km)", s_enu)
        print(r_enu-s_enu)
        
        #r_mask = self.mypinn_t.large_model.is_within_domain(r_enu)
        #s_mask = self.mypinn_t.large_model.is_within_domain(s_enu)
        r_mask = self.is_within_domain(r_lon, r_lat)
        s_mask = self.is_within_domain(s_lon, s_lat, s_h)
        

        # Create errcodelist where each element corresponds to each point in r_enu
        # 0 = within domain, 1 = outside domain
        errcodelist = (r_mask).to(torch.int32).numpy() & (s_mask).to(torch.int32).numpy()   
        
        tt = self.mypinn_t.traveltime_pred_obs_reciprocity(r_enu, s_enu)

        tt_np = tt.numpy()

        return tt_np, errcodelist.tolist()


    def convert_receiver_to_enu(self, r_lon, r_lat):
        # inputted #elevation is in km
        
        r_geo = torch.from_numpy(self.myegm.interpolate(r_lat.detach().numpy(), r_lon.detach().numpy()))
        r_h   = self.gebco_nearestneighbor(r_lon, r_lat).clone().detach()#.to(torch.double)
#        print("r_h(m)", r_h)

        if self.device != torch.device("cpu"):
            torch.set_default_tensor_type(cuda_ttype)
            r_lon = r_lon.to(self.device)
            r_lat = r_lat.to(self.device)
            r_h   = r_h.to(self.device)

        r_e, r_n, r_u = pymap3d_torch.geodetic2enu(r_lat, r_lon, r_h+r_geo, self.lat0, self.lon0, self.h0)
        r_enu = torch.vstack((torch.vstack((r_e, r_n)), r_u)).T*1e-3 #m=>km
        r_enu = rotate_translate_matmul(r_enu, self.cart_corners[3], self.rot)
        return r_enu
    
    def convert_source_to_enu(self, s_lon, s_lat, s_h_km):
        # inputted #elevation is in km
        s_h   = s_h_km*1e+3 #km=>m
        
        geoid = self.myegm.interpolate(s_lat.detach().numpy(), s_lon.detach().numpy())
        if isinstance(geoid, np.ndarray):
            s_geo = torch.from_numpy(geoid)
        else:
            s_geo = torch.tensor([geoid])
        #s_geo = torch.from_numpy(self.myegm.interpolate(s_lat.detach().numpy(), s_lon.detach().numpy()))

        if self.device != torch.device("cpu"):
            torch.set_default_tensor_type(cuda_ttype)
            s_lon = s_lon.to(self.device)
            s_lat = s_lat.to(self.device)
            s_h   = s_h.to(self.device)

        s_e, s_n, s_u = pymap3d_torch.geodetic2enu(s_lat, s_lon, s_h+s_geo, self.lat0, self.lon0, self.h0)
        s_enu = torch.vstack((torch.vstack((s_e, s_n)), s_u)).T*1e-3 #m=>km
        s_enu = rotate_translate_matmul(s_enu, self.cart_corners[3], self.rot)
        return s_enu
    
    def inference_torch(self, s_lon, s_lat, s_h_km, r_lon, r_lat):      
        """
        Perform travel time inference using PyTorch tensors.
        
        Args:
            s_lon, s_lat: Source longitude and latitude
            s_h_km: Source depth in km
            r_lon, r_lat: Receiver longitude and latitude
            
        Returns:
            tuple: (travel_times, error_codes)
        """        
        # inputted #elevation is in km
        s_h   = s_h_km*1e+3 #km=>m
        r_geo = torch.from_numpy(self.myegm.interpolate(r_lat.detach().numpy(), r_lon.detach().numpy()))
        s_geo = torch.from_numpy(self.myegm.interpolate(s_lat.detach().numpy(), s_lon.detach().numpy()))
        r_h = self.gebco_nearestneighbor(r_lon, r_lat).clone().detach()

        if self.device != torch.device("cpu"):
            torch.set_default_tensor_type(cuda_ttype)
            s_lon = s_lon.to(self.device)
            s_lat = s_lat.to(self.device)
            s_h = s_h.to(self.device)
            r_lon = r_lon.to(self.device)
            r_lat = r_lat.to(self.device)
            r_h = r_h.to(self.device)

        s_e, s_n, s_u = pymap3d_torch.geodetic2enu(s_lat, s_lon, s_h+s_geo, self.lat0, self.lon0, self.h0)
        r_e, r_n, r_u = pymap3d_torch.geodetic2enu(r_lat, r_lon, r_h+r_geo, self.lat0, self.lon0, self.h0)
        s_enu = torch.vstack((torch.vstack((s_e, s_n)), s_u)).T*1e-3 #m=>km
        r_enu = torch.vstack((torch.vstack((r_e, r_n)), r_u)).T*1e-3 #m=>km
        s_enu = rotate_translate_matmul(s_enu, self.cart_corners[3], self.rot)
        r_enu = rotate_translate_matmul(r_enu, self.cart_corners[3], self.rot)

        r_mask = self.is_within_domain(r_lon, r_lat)
        s_mask = self.is_within_domain(s_lon, s_lat, s_h)
        
        # errcodelist: 0 if both r_mask and s_mask are True, 1 otherwise, for each element
        errcodelist = (~(r_mask & s_mask)).to(torch.int32)    
        
        tt = self.mypinn_t.traveltime_pred_obs_reciprocity(r_enu, s_enu)
        return tt, errcodelist.tolist()
