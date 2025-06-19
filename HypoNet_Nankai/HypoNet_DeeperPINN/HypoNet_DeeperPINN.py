import torch
import HypoNet_Nankai.HypoNet_DeeperPINN.pymap3d_torch as pymap3d_torch
import numpy as np
import HypoNet_Nankai.HypoNet_DeeperPINN.geodetic as geodetic
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#from math import cos, sin
#from numba import jit
def rotate_translate_matmul(coor, origin, rot):
    """Apply rotation and translation to coordinates."""
    coor_mod = torch.empty_like(coor)
    coor_mod[:,2] = coor[:,2] 
    coor_mod[:,0:2] = torch.subtract(coor[:,0:2],origin)
#    for i in range(len(coor)):
#        coor_mod[i,0:2] = rot@coor_mod[i,0:2]
    coor_mod[:,0:2] = (rot@coor_mod[:,0:2].T).T
    return coor_mod

class PINNtomo_param:
    def read(self, inputdir):
#        if im == 0:
        with open(BASE_DIR+"/"+inputdir+"/setting") as f:
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
        """
        dummy function for S-wave
        """


    def inference_torch(self, s_lon, s_lat, s_h_km, r_lon, r_lat):
        """
        dummy function for S-wave
        """
        tt = torch.tensor([])
        errcodelist = torch.tensor([])

        return tt, errcodelist.tolist()
