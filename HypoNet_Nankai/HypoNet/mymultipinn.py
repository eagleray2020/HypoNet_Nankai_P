import numpy as np
import time
import random
import torch
import sys
import copy
import math
import HypoNet_Nankai.HypoNet.mypinn_FF_gpu_3D as mypinn
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def consecutive(data, base_in, stepsize=0):
#    print(data.get_device(), base.get_device())
    if  base_in.get_device() == 0:
        base = base_in.to("cpu")
    else:
        base = base_in
    return torch.tensor_split(data, torch.where(torch.diff(base) != stepsize)[0]+1)

class MyMultiPINN():
    def __init__(self, dirn, fact_name_l, nhl_l, nnt_l, nmff_l, FFsigma):

        def convert_1d_to_2d(l, cols):
            return [l[i:i + cols] for i in range(0, len(l), cols)]

        coor = np.genfromtxt(BASE_DIR+"/"+dirn+"/coor_minmax.txt")
        xmin = coor[0,0]
        xmax = coor[0,1]
        ymin = coor[1,0]
        ymax = coor[1,1]
        zmin = coor[2,0]
        zmax = coor[2,1]
        vmin = coor[3,0]
        vmax = coor[3,1]
        self.large_model = mypinn.MyPINN3D_t(fact_name_l, nhl_l, nnt_l, nmff_l, FFsigma)
        self.large_model.set_coor_minmax(xmin, xmax, ymin, ymax, zmin, zmax)
        self.large_model.set_v_minmax_from_scalar(vmin, vmax)

        div_setting = np.genfromtxt(BASE_DIR+"/"+dirn+"/divide_setting", comments="#")
        nsx, nsy = div_setting[0,0].astype(np.int32), div_setting[0,1].astype(np.int32)
        ndx, ndy = div_setting[1,0].astype(np.int32), div_setting[1,1].astype(np.int32)
        self.ndx = ndx; self.ndy = ndy
        """
        self.locxmin_list = np.zeros((ndy, ndx))
        self.locxmax_list = np.zeros((ndy, ndx))
        self.locymin_list = np.zeros((ndy, ndx))
        self.locymax_list = np.zeros((ndy, ndx))
        """
        self.locxmin_list = torch.zeros((ndy, ndx))
        self.locxmax_list = torch.zeros((ndy, ndx))
        self.locymin_list = torch.zeros((ndy, ndx))
        self.locymax_list = torch.zeros((ndy, ndx))
        self.models = []
        for iy in range(ndy):
            for ix in range(ndx):
                input_dir = f"dir_x{ix+1}_y{iy+1}/input/"
                input_dir = dirn+"/"+input_dir
                with open(BASE_DIR+"/"+input_dir+"/setting") as f:
                    s = f.readline()
                    s = f.readline()
                    items = s.split()
                    nhlt = int(items[0])
                    nnt = int(items[1])
                    fact_name = items[2]
                    nmff = len(items)-3
                    FFsigma = []
                    for i in range(nmff):
                        FFsigma.append(float(items[i+3]))
                mypinn_t = mypinn.MyPINN3D_t(fact_name, nhlt, nnt, nmff, FFsigma)
                coor = np.genfromtxt(BASE_DIR+"/"+input_dir+"/coor_minmax.txt")
                xminm = coor[0,0]
                xmaxm = coor[0,1]
                yminm = coor[1,0]
                ymaxm = coor[1,1]
                zminm = coor[2,0]
                zmaxm = coor[2,1]
                vminm = coor[3,0]
                vmaxm = coor[3,1]
                mypinn_t.set_coor_minmax(xminm, xmaxm, yminm, ymaxm, zminm, zmaxm)
                mypinn_t.set_v_minmax_from_scalar(vmin, vmax)
                self.models.append(copy.deepcopy(mypinn_t))
                self.locxmin_list[iy, ix] = xminm
                self.locxmax_list[iy, ix] = xmaxm
                self.locymin_list[iy, ix] = yminm
                self.locymax_list[iy, ix] = ymaxm

        self.models = convert_1d_to_2d(self.models, ndx)

        dx = (xmax-xmin)/nsx
        dy = (ymax-ymin)/nsy

        self.griddx = (xmax-dx-xmin)/(ndx-1)
        self.griddy = (ymax-dy-ymin)/(ndy-1)

        self.xmin_grid = xmin+dx*0.5
        self.ymin_grid = ymin+dy*0.5

        #print(self.xmin_grid, self.griddx, ndx)
        #print(self.ymin_grid, self.griddy, ndy)



    def input_models(self, dirn, device):
        path = BASE_DIR+"/"+dirn+"/mypinn_t_state_dict.pt"
        self.large_model.load_state_dict(torch.load(path,map_location=device))

        ndx = self.ndx
        ndy = self.ndy
        for iy in range(ndy):
            for ix in range(ndx):
                input_dir = f"dir_x{ix+1}_y{iy+1}/output/"
                path = BASE_DIR+"/"+dirn+"/"+input_dir+"/mypinn_t_state_dict.pt"
                self.models[iy][ix].load_state_dict(torch.load(path, map_location=device))

    def nearest_small_domain(self, xs, x):
        ndx = self.ndx
        ndy = self.ndy
#        print(xs)
#        print(self.xmin_grid, self.griddx)
#        print(self.ymin_grid, self.griddy)
#        id_small_domain = torch.empty(len(x), dtype=torch.int32)
        id_small_domain_buf = torch.empty((len(xs),2), dtype=torch.int32)
        id_small_domain_buf[:,0] = torch.round((xs[:,0]-self.xmin_grid)/self.griddx).to(torch.int32)
        id_small_domain_buf[:,0] = torch.minimum(torch.maximum(id_small_domain_buf[:,0], torch.zeros(len(xs))), torch.ones(len(xs))*(ndx-1))
        id_small_domain_buf[:,1] = torch.round((xs[:,1]-self.ymin_grid)/self.griddy).to(torch.int32)
        id_small_domain_buf[:,1] = torch.minimum(torch.maximum(id_small_domain_buf[:,1], torch.zeros(len(xs))), torch.ones(len(xs))*(ndy-1))
        id_small_domain = id_small_domain_buf[:,0]+id_small_domain_buf[:,1]*ndx
 #       print(self.locxmin_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]])
 #       print(self.locxmax_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]])
 #       print(self.locymin_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]])
 #       print(self.locymax_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]])
 #       print(x)
#        print(id_small_domain_buf)
#        print((xs[:,0]-self.xmin_grid)/self.griddx)
        ind = torch.where((x[:,0]<self.locxmin_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]]) |
                          (x[:,0]>self.locxmax_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]]) |
                          (x[:,1]<self.locymin_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]]) |
                          (x[:,1]>self.locymax_list[id_small_domain_buf[:,1], id_small_domain_buf[:,0]]))
        id_small_domain[ind] = -1
#        print(ind)

        return id_small_domain.clone().detach()

    def pred_t(self, x, xs):
        ndx = self.ndx
        ndy = self.ndy
#        start = time.time()
        id_small_domain = self.nearest_small_domain(x, xs)
        id_small_domain_sorted, ind = torch.sort(id_small_domain)
#        id_small_domain_sorted = id_small_domain.sort()
#        ind = id_small_domain.argsort()
        id_small_domain_sorted_splitted = consecutive(id_small_domain_sorted, id_small_domain_sorted)
        ind_splitted = consecutive(ind, id_small_domain_sorted)

#        elapsed = time.time()-start
#        print(i_s, "nearsest_small_domain took", elapsed, "s")
#        np.savetxt("output/id.csv", torch.vstack((x.T, id_small_domain.to(torch.float32).T)).T.to("cpu"), delimiter=",")
        t1 = 0.
        t = torch.empty((len(x), 1))
        start = time.time()
        for id_e, ind_e in zip(id_small_domain_sorted_splitted, ind_splitted):
#            print(id_e, ind_e)
#            print(id_e.dtype, ind_e.dtype)
            if len(id_e) > 0:
                if id_e[0] == -1:
#                    print(x[ind_e], xs[ind_e])
                    t[ind_e] = self.large_model.pred_t(x[ind_e], xs[ind_e])
                else:
                    ix = id_e[0].item()%ndx
                    iy = int(id_e[0].item()/ndx)
                    t[ind_e]  = self.models[iy][ix].pred_t(x[ind_e], xs[ind_e])
#                    print(id_e[0], ix, iy, len(id_e))#, x[ind_e], xs[ind_e])
        t1 += time.time()-start

        """
        for iy in range(ndy):
            for ix in range(ndx):
                xmin = self.locxmin_list[iy, ix]
                xmax = self.locxmax_list[iy, ix]
                ymin = self.locymin_list[iy, ix]
                ymax = self.locymax_list[iy, ix]
                ind  = torch.where((id_small_domain[:,0]==ix) &
                                   (id_small_domain[:,1]==iy) &
                                   (xs[:,0]>=xmin) & (xs[:,0]<=xmax) &
                                   (xs[:,1]>=ymin) & (xs[:,1]<=ymax))
                ind2 = torch.where((id_small_domain[:,0]==ix) &
                                   (id_small_domain[:,1]==iy) &
                                   ((xs[:,0]<xmin) | (xs[:,0]>xmax) |
                                    (xs[:,1]<ymin) | (xs[:,1]>ymax)))
                start = time.time()
                t[ind]  = self.models[iy][ix].pred_t(x[ind], xs[ind])
                t[ind2] = self.large_model.pred_t(x[ind2], xs[ind2])
        #t = self.large_model.pred_t(x, xs)
                t1 += time.time()-start
        """
#        print("inference took", t1, "s")

        return t

    def traveltime_pred_obs_reciprocity(self, x, xs):
        ndata = len(x)
        t0 = ((x[:,0]-xs[:,0])**2+(x[:,1]-xs[:,1])**2+(x[:,2]-xs[:,2])**2)**0.5
        tau = self.pred_t(xs, x)
        t = tau[:, 0]*t0
        return t

