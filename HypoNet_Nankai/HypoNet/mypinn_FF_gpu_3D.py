import numpy as np
import time
import random
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import scipy.io 
import copy
import scipy.sparse.linalg
#import torch_optimizer as optim
import math
import matplotlib.pyplot as plt
import time 
import subprocess

#seed = 0
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
def gauss(x):
    return torch.exp(-x**2)

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
            x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]


class MyPINN_t(torch.nn.Module):
    def __init__(self, fact_name, nhl, nnt, nmff, FFsigma=None):
        super(MyPINN_t, self).__init__()
        self.nhl = nhl
        self.nnt = nnt
        self.nmff = nmff

        if FFsigma != None:
            if math.isclose(FFsigma[0], 0.):
                self.bool_ff1 = False
            else:
                self.bool_ff1 = True
            if nmff >= 2:
                if math.isclose(FFsigma[1], 0.):
                    self.bool_ff2 = False
                else:
                    self.bool_ff2 = True
            if nmff >= 3:
                if math.isclose(FFsigma[2], 0.):
                    self.bool_ff3 = False
                else:
                    self.bool_ff3 = True
        else:
            self.bool_ff1 = True
            self.bool_ff2 = True
            self.bool_ff3 = True

        if self.bool_ff1 == True:
            self.ff1 = torch.nn.Linear(6, int(self.nnt/2), bias=False)
        else:
            self.ln1 = torch.nn.Linear(6, self.nnt)

        if nmff >= 2:
            if self.bool_ff2 == True:
                self.ff2 = torch.nn.Linear(6, int(self.nnt/2), bias=False)
            else:
                self.ln2 = torch.nn.Linear(6, self.nnt)

        if nmff >= 3:
            if self.bool_ff3 == True:
                self.ff3 = torch.nn.Linear(6, int(self.nnt/2), bias=False)
            else:
                self.ln3 = torch.nn.Linear(6, self.nnt)

        if nmff >= 4:
            print("Error: nmff >= 4 is not supported now")
            sys.exit()

        self.fct1 = torch.nn.Linear(self.nnt, self.nnt)
        self.fct2 = torch.nn.Linear(self.nnt, self.nnt)
        
        if self.nhl == 20:
            self.fct3 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct4 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct5 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct6 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct7 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct8 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct9 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct10 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct11 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct12 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct13 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct14 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct15 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct16 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct17 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct18 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct19 = torch.nn.Linear(self.nnt, self.nnt)
            self.fct20 = torch.nn.Linear(self.nnt, self.nnt)
        else:
            print("Error: nhl == ", self.nhl, " is not supported now")

        self.outl = torch.nn.Linear(self.nnt*self.nmff, 1)

        if fact_name == "swish":
            self.actv = torch.nn.SiLU()
        elif fact_name == "gauss":
            self.actv = gauss
        elif fact_name == "tanh":
            self.actv = torch.nn.Tanh()
        else:
            print("activation function ", fact_name, " is not valid")
            print(fact_name, nhl, nnt)
            sys.exit()

        self.loss_function = torch.nn.MSELoss(reduction ='sum')
        self.loss_function_mean = torch.nn.MSELoss(reduction ='mean')
        
    #def pred_t_nobn(self, x):
    def set_coor_minmax(self, xmin, xmax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.zmin = zmin
        self.zmax = zmax
    
    def normalize_coor_four(self, x):
    #following Rasht-Behesht et al. (2022)
        x_out = x.clone().detach()
        x_out[:,0] = 2.*x[:,0]/(self.xmax-self.xmin)-1.
        x_out[:,1] = 2.*x[:,1]/(self.zmax-self.zmin)-1.
        x_out[:,2] = 2.*x[:,2]/(self.xmax-self.xmin)-1.
        x_out[:,3] = 2.*x[:,3]/(self.zmax-self.zmin)-1.
        return x_out
    
    def normalize_coor_four_xz_semi3D(self, x):
    #following Rasht-Behesht et al. (2022)
#        x_out = x.clone().detach()
        x_out = torch.zeros(len(x),6)
        x_out[:,0] = 2.*x[:,0]/(self.xmax-self.xmin)-1.
        x_out[:,2] = 2.*x[:,1]/(self.zmax-self.zmin)-1.
        x_out[:,3] = 2.*x[:,2]/(self.xmax-self.xmin)-1.
        x_out[:,5] = 2.*x[:,3]/(self.zmax-self.zmin)-1.
        return x_out

#    def pred_t(self, x):
    def pred_arctanhveff(self, x):
        x = self.normalize_coor_four_xz_semi3D(x)
#        x = self.fct1(x)
        x = self.fct1(x)
#        x = torch.cat((torch.sin(x),torch.cos(x)), axis=1)
        x = torch.cat((torch.sin(2.*np.pi*x),torch.cos(2.*np.pi*x)), axis=1)
#        x = self.actv(x)
        x = self.fct2(x)
        x = self.actv(x)
        x = self.fct3(x)
        if self.nhl == 20:
            x = self.actv(x)
            x = self.fct4(x)
            x = self.actv(x)
            x = self.fct5(x)
            x = self.actv(x)
            x = self.fct6(x)
            x = self.actv(x)
            x = self.fct7(x)
            x = self.actv(x)
            x = self.fct8(x)
            x = self.actv(x)
            x = self.fct9(x)
            x = self.actv(x)
            x = self.fct10(x)
            x = self.actv(x)
            x = self.fct11(x)
            x = self.actv(x)
            x = self.fct12(x)
            x = self.actv(x)
            x = self.fct13(x)
            x = self.actv(x)
            x = self.fct14(x)
            x = self.actv(x)
            x = self.fct15(x)
            x = self.actv(x)
            x = self.fct16(x)
            x = self.actv(x)
            x = self.fct17(x)
            x = self.actv(x)
            x = self.fct18(x)
            x = self.actv(x)
            x = self.fct19(x)
            x = self.actv(x)
            x = self.fct20(x)
            x = self.actv(x)
            x = self.fct21(x)

        return x
            
#    def pred_t(self, x):
#        x = self.pred_arctanhveff(x)
    def pred_t(self, x, xs):
        ndata = len(x)
        xj = torch.empty((ndata,4))
        xjr = torch.empty((ndata,4))
        xj[:, 0] = x[:, 0]
        xj[:, 1] = x[:, 1]
        xj[:, 2] = xs[:, 0]
        xj[:, 3] = xs[:, 1]
        xjr[:, 0] = xs[:, 0]
        xjr[:, 1] = xs[:, 1]
        xjr[:, 2] = x[:, 0]
        xjr[:, 3] = x[:, 1]
        x = 0.5*self.pred_arctanhveff(xj)+0.5*self.pred_arctanhveff(xjr)
        #test 
#        x = x*2.-1.
#        x = x-0.5
        x = torch.tanh(x)
        x = (x+1.)/2.*(self.vtotmax*1.1-self.vtotmin*0.9)+self.vtotmin*0.9
#        x = (x+1.)/2.*(self.vtotmax-self.vtotmin)+self.vtotmin
        x = 1./x
#        x = torch.tanh(x)
#        x = (x+1.)/2.*(self.taumax-self.taumin)+self.taumin
        return x
 
    def pred_T(self, x, xs):
        ndata = len(x)
        t0 = (((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5)/(self.pred_v(xs))[:, 0]
        tau = self.pred_t(x, xs)
        T = tau[:, 0]*t0
        return T
        
    def num_of_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, xpde, xobs, xsource, xs_pde, xs_obs, xs_source):
        p = self.forward_misfit(xpde, xobs, xsource, xs_pde, xs_obs, xs_source)
        q = self.prior()
        return torch.cat((p, q))
 
    def prior(self):
        if self.hp_prior < 0.:
            return  torch.zeros(self.num_of_parameters())
        else:
            return torch.cat([param.view(-1) for param in self.parameters()])/(2**0.5*self.hp_prior)

    def forward_misfit(self, xpde, xobs, xsource, xs_pde, xs_obs, xs_source):
        return  torch.cat((torch.cat((self.forward_pde(xpde, xs_pde), self.forward_obs(xobs, xs_obs)), 0),\
                           self.forward_source(xsource, xs_source)), 0)

    #def forward_eik(self, xpde, xsource, xs_pde, xs_source, v):
    #    return  torch.cat((self.forward_pde(xpde, xs_pde, v), self.forward_source(xsource, xs_source)), 0)

    def forward_pde(self, x, xs, v):
        ndata = len(x)
        xj = torch.empty((ndata,4))
        xj[:, 0] = x[:, 0]
        xj[:, 1] = x[:, 1]
        xj[:, 2] = xs[:, 0]
        xj[:, 3] = xs[:, 1]
        tau = self.pred_t(xj)
           
        t0 = ((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5
         
        tau_x = torch.autograd.grad(tau[:, 0], x, torch.ones(ndata), retain_graph=True, create_graph=True)
        t0_x  = torch.autograd.grad(t0, x, torch.ones(ndata), retain_graph=True, create_graph=True)   
        
        #self.v_pred = (self.pred_v(x))[:, 0]
        
        p = (t0**2*(tau_x[0][:, 0]**2+tau_x[0][:, 1]**2)+tau[:, 0]**2*(t0_x[0][:, 0]**2+t0_x[0][:, 1]**2)+ \
          + 2.*t0*tau[:, 0]*(tau_x[0][:, 0]*t0_x[0][:, 0]+tau_x[0][:, 1]*t0_x[0][:, 1]))**(-0.5)-v
#        p = (t0**2*(tau_x[0][:, 0]**2+tau_x[0][:, 1]**2)+tau[:, 0]**2*(t0_x[0][:, 0]**2+t0_x[0][:, 1]**2)+ \
#          + 2.*t0*tau[:, 0]*(tau_x[0][:, 0]*t0_x[0][:, 0]+tau_x[0][:, 1]*t0_x[0][:, 1]))**(-1)-v**2
        p /= (2**0.5*self.hp_pde)
        
        #print("p", p)
        return p

    def forward_pde_nohp(self, x, xs, v):
        ndata = len(x)
        tau = self.pred_t(x, xs)
        t0 = ((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5
        T = tau[:, 0]*t0
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        p = (T_x[:,0]**2+T_x[:,1]**2)**(-0.5)-v
        p /= (len(v)**0.5)
        return p

    def forward_pde_nohp_nonormalization(self, x, xs, v):
        ndata = len(x)
        tau = self.pred_t(x, xs)
        t0 = ((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5
        T = tau[:, 0]*t0
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        p = (T_x[:,0]**2+T_x[:,1]**2)**(-0.5)-v
        return p
  
    def forward_source_nohp(self, xs_s, vs):
        return (1./self.pred_t(xs_s,xs_s)[:, 0]-vs)/(len(vs)**0.5)


    def forward_nohp(self, x, xs, xs_s, v, vs):
        p = self.forward_pde_nohp(x, xs, v)
        #return p
        q = self.forward_source_nohp(xs_s, vs)
        return torch.cat((p, q))

    
    def forward_obs(self, x, xs):
        ndata = len(x)
        xj = torch.empty((ndata,4))
        xj[:, 0] = x[:, 0]
        xj[:, 1] = x[:, 1]
        xj[:, 2] = xs[:, 0]
        xj[:, 3] = xs[:, 1]
        t0 = (((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5)
        tau = self.pred_t(xj)
        t = tau[:, 0]*t0       
        t /= (2**0.5*self.hp_obs)
            
        return t
    
 
    def set_num_points(self, ns, nobs):
        self.ns = ns
        self.nobs = nobs        
   
    def set_t_minmax(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
 
    def set_tau_minmax(self, taumin, taumax):
        self.taumin = taumin
        self.taumax = taumax 

    def set_v_minmax(self, fvmean, fptbvmin, fptbvmax, Xt, Zt):
        x = Xt.ravel()
        z = Zt.ravel()
        x = torch.transpose(torch.vstack([x, z]), 0, 1).clone()
        self.vtotmin = torch.min(fvmean(x)+fptbvmin(x))
        self.vtotmax = torch.max(fvmean(x)+fptbvmax(x))

    def set_v_minmax_from_scalar(self, vtotmin, vtotmax):
        self.vtotmin = vtotmin
        self.vtotmax = vtotmax

    def set_hyper_parameter(self, hp_pde, hp_source, hp_obs, hp_prior, hp_fprior1, hp_fprior2, hp_fprior3):
        self.hp_pde = torch.tensor(hp_pde)
        self.hp_source = torch.tensor(hp_source)
        self.hp_obs = torch.tensor(hp_obs)
        self.hp_prior = torch.tensor(hp_prior)
        #if self.hp_prior < 0.:
        #    print("hp_prior is input as negative, but is set to be inifinite instead")
        self.hp_fprior1 = torch.tensor(hp_fprior1)
        self.hp_fprior2 = torch.tensor(hp_fprior2)
        self.hp_fprior3 = torch.tensor(hp_fprior3)



    def traveltime_pred_obs(self, x, xs):
        ndata = len(x)
        
        t0 = (((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5)
        
        tau = self.pred_t(x, xs)
        t = tau[:, 0]*t0
        
        return t
    
    def traveltime_pred_grid(self, X, Z, xs):
        x = X.ravel()
        z = Z.ravel()
        x = torch.transpose(torch.vstack([x, z]), 0, 1).clone()
        ndata = len(x)
        xs_list = torch.ones(ndata,2)
        xs_list[:, 0] *= xs[0]
        xs_list[:, 1] *= xs[1]
        tau = self.pred_t(x,xs_list)
        #xpde = x[:, 0:2]
        #xs = x[:, 2:4]
        t0 = (((x[:, 0]-xs[0])**2+(x[:, 1]-xs[1])**2)**0.5)
        t = tau[:, 0]*t0
        
        t = (t).reshape(len(X), len(X[0, :])).detach()#.numpy().copy()
        
        return t

    def velocity_pred_by_pde(self, X, Z, xs):
        x = X.ravel()
        z = Z.ravel()
        x = torch.transpose(torch.vstack([x, z]), 0, 1).clone()
        
        ndata = len(x)
        xslist = torch.empty((ndata,2))
        xslist[:, 0] = xs[0]
        xslist[:, 1] = xs[1]
        tau = self.pred_t(x, xslist)
        #xpde = x[:, 0:2]
        #xs = x[:, 2:4]
        t0 = (((x[:, 0]-xs[0])**2+(x[:, 1]-xs[1])**2)**0.5)/(self.pred_v(xs))
        #print((self.pred_v(xs))[:, 0])
        #print(t0)
         
        tau_x = torch.autograd.grad(tau[:, 0], x, torch.ones(ndata), retain_graph=True, create_graph=True)
        t0_x  = torch.autograd.grad(t0, x, torch.ones(ndata), retain_graph=True, create_graph=True)   
        #tau_x = torch.autograd.grad(tau[:, 0], x, torch.ones(ndata), retain_graph=True)
        #t0_x  = torch.autograd.grad(t0, x, torch.ones(ndata), retain_graph=True)
        
        v = (t0**2*(tau_x[0][:, 0]**2+tau_x[0][:, 1]**2)+tau[:, 0]**2*(t0_x[0][:, 0]**2+t0_x[0][:, 1]**2)+ \
          + 2.*t0*tau[:, 0]*(tau_x[0][:, 0]*t0_x[0][:, 0]+tau_x[0][:, 1]*t0_x[0][:, 1]))**(-0.5)
                
        v = (v).reshape(len(X), len(X[0, :])).detach()#.numpy().copy()
        #print("p", p)
        return v
    

    def set_param_serial(self):
        nparam = self.num_of_parameters()
        self.nid_param_serial = np.zeros(nparam, dtype=np.int32)
        self.id1_param_serial = np.full(nparam, -1, dtype=np.int32)        
        self.id2_param_serial = np.full(nparam, -1, dtype=np.int32)        
        self.name_param_serial = []
        istart_list = np.array([])
        
        state_dict = self.state_dict()
        
        istart = 0
        iend = 0
        for name, param in self.named_parameters():
            
            iend += sum(p.numel() for p in param)
            for i in range(istart, iend):
                (self.name_param_serial).append(name)
                istart_list = np.append(istart_list, int(istart))
            istart = iend            
        istart_list = istart_list.astype(np.int32)
            
        for i in range(nparam):
            nid = (state_dict[self.name_param_serial[i]]).dim()
            self.nid_param_serial[i] = nid
            if nid == 1:
                self.id1_param_serial[i] = i - istart_list[i]
            else:            
                a, b = (state_dict[self.name_param_serial[i]]).size()
                self.id1_param_serial[i] = int((i-istart_list[i])/b)
                self.id2_param_serial[i] = (i-istart_list[i])%b
            
    
    def decompose_vector_like_parameters(self, v):
        if len(v) != self.num_of_parameters():
            print("length of input vector is not consistent with nn parameters")
            return
        
        head = 0
        v_dcomp = torch.tensor([])
        v_dcomp = []
        for name, param in self.named_parameters():
            leng = sum(p.numel() for p in param)
            if param.dim() == 1:
                v_dcomp.append(v[head:head+leng])
            else:
                a, b = param.size()
                #print(a, b)
                v_dcomp.append(torch.reshape(v[head:head+leng], (a, b)))
            head += leng
            
        return v_dcomp
    
    def forwardf_misfit(self, v, gT, Tobs):
        return  torch.cat((self.forwardf_pde(v, gT), self.forwardf_obs(Tobs)), 0)         
            
    def forwardf_pde(self, v, gT):
        p = (gT[:, 0]**2+gT[:, 1]**2)**(-0.5)-v[:]
        p /= (2**0.5*self.hp_pde)
        #print("p", p)
        return p
    
    def forwardf_source(self, Ts):
        return Ts/(2**0.5*self.hp_source)
    
    def forwardf_obs(self, Tobs, std_obs):
        #t = Tobs/(2**0.5*self.hp_obs)
        t = Tobs/(2**0.5*std_obs)
        return t 
            
    def pred_gT(self, xpde, xs_pde):
        ndata = len(xpde)
        tau = self.pred_t(xpde, xs_pde)
        t0 = ((xpde[:, 0]-xs_pde[:, 0])**2+(xpde[:, 1]-xs_pde[:, 1])**2)**0.5
        T = tau[:, 0]*t0
        gT = torch.autograd.grad(T.sum(), xpde, create_graph=True)[0]
        
        return gT

    def pred_Tobs(self, xobs, xs_obs):
        ndata = len(xobs)
        tau = self.pred_t(xobs, xs_obs)
        t0 = ((xobs[:, 0]-xs_obs[:, 0])**2+(xobs[:, 1]-xs_obs[:, 1])**2)**0.5
        Tobs = tau[:, 0]*t0
   
        return Tobs

    def calc_rms(self, x, xs, y):
        ndata = len(x)
        t0 = (((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5)
        tau = self.pred_t(x, xs)
        t = tau[:, 0]*t0       
        
        return self.loss_function_mean(y.ravel(), t.ravel())**0.5

    def loss_obs(self, x_obs, xs_obs, y_obs, std_obs):
        Tobs = self.pred_Tobs(x_obs, xs_obs)
        y_obs_scaled = (y_obs.reshape(len(y_obs))/(2**0.5*(std_obs))).clone().detach()
        outp = self.forwardf_obs(Tobs, std_obs)
        return self.loss_function(outp, y_obs_scaled)

    def initialization_kaiming(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def initialization_Xavier(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def initialization_FFswish(self, FFsigma):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
#                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
            
        if self.bool_ff1 == True:
            torch.nn.init.normal_(self.ff1.weight,mean=0.,std=FFsigma[0])
        if self.nmff >= 2:
            if self.bool_ff2 == True:
                torch.nn.init.normal_(self.ff2.weight,mean=0.,std=FFsigma[1])
        if self.nmff >= 3:
            if self.bool_ff3 == True:
                torch.nn.init.normal_(self.ff3.weight,mean=0.,std=FFsigma[2])

    def reset_all_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

class matvec_dvdw_dvdwT():
    def __init__(self, v, pinn_v):
        self.v = v
        self.pinn_v = pinn_v

    def matvec(self, vec):
        dummy = torch.randn(len(self.v), requires_grad=True)
        dvdw_dummy = torch.cat([param.view(-1) for param in torch.autograd.grad(self.v @ dummy, self.pinn_v.parameters(), retain_graph=True, create_graph=True)])
        dvdwT_vec = torch.cat([param.view(-1) for param in torch.autograd.grad(dvdw_dummy @ vec, dummy, retain_graph=True, create_graph=True)])
        dvdw_dvdwT_vec = torch.cat([param.view(-1) for param in torch.autograd.grad(self.v @ dvdwT_vec, self.pinn_v.parameters(), retain_graph=True)])
        print(dvdw_dvdwT_vec)
        return dvdw_dvdwT_vec

class MyPINN3D_t(MyPINN_t): 
    def set_coor_minmax(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def normalize_coor_six(self, x):
    #following Rasht-Behesht et al. (2022)
        epsilon = 1e-4
        x_out = torch.zeros(len(x),6)
        x_out[:,0] = 2.*(x[:,0]-self.xmin)/(self.xmax-self.xmin+epsilon)-1.
        x_out[:,1] = 2.*(x[:,1]-self.ymin)/(self.ymax-self.ymin+epsilon)-1.
        x_out[:,2] = 2.*(x[:,2]-self.zmin)/(self.zmax-self.zmin+epsilon)-1.
        x_out[:,3] = 2.*(x[:,3]-self.xmin)/(self.xmax-self.xmin+epsilon)-1.
        x_out[:,4] = 2.*(x[:,4]-self.ymin)/(self.ymax-self.ymin+epsilon)-1.
        x_out[:,5] = 2.*(x[:,5]-self.zmin)/(self.zmax-self.zmin+epsilon)-1.
        return x_out

    def pred_arctanhveff(self, x):
        x_in = self.normalize_coor_six(x)
        for i in range(self.nmff):
            if i == 0:
                if self.bool_ff1 == True:
                    x = self.ff1(x_in)
                    x = torch.cat((torch.sin(2.*np.pi*x),torch.cos(2.*np.pi*x)), axis=1)
                else:
                    x = self.ln1(x_in)
                    x = self.actv(x)
            elif i == 1:
                if self.bool_ff2 == True:
                    x = self.ff2(x_in)
                    x = torch.cat((torch.sin(2.*np.pi*x),torch.cos(2.*np.pi*x)), axis=1)
                else:
                    x = self.ln2(x_in)
                    x = self.actv(x)
            elif i == 2:
                if self.bool_ff3 == True:
                    x = self.ff3(x_in)
                    x = torch.cat((torch.sin(2.*np.pi*x),torch.cos(2.*np.pi*x)), axis=1)
                else:
                    x = self.ln3(x_in)
                    x = self.actv(x)

            x = self.fct1(x)
            x = self.actv(x)
            x = self.fct2(x)
            x = self.actv(x)
            x = self.fct3(x)
            if self.nhl == 20:
                x = self.actv(x)
                x = self.fct4(x)
                x = self.actv(x)
                x = self.fct5(x)
                x = self.actv(x)
                x = self.fct6(x)
                x = self.actv(x)
                x = self.fct7(x)
                x = self.actv(x)
                x = self.fct8(x)
                x = self.actv(x)
                x = self.fct9(x)
                x = self.actv(x)
                x = self.fct10(x)
                x = self.actv(x)
                x = self.fct11(x)
                x = self.actv(x)
                x = self.fct12(x)
                x = self.actv(x)
                x = self.fct13(x)
                x = self.actv(x)
                x = self.fct14(x)
                x = self.actv(x)
                x = self.fct15(x)
                x = self.actv(x)
                x = self.fct16(x)
                x = self.actv(x)
                x = self.fct17(x)
                x = self.actv(x)
                x = self.fct18(x)
                x = self.actv(x)
                x = self.fct19(x)
                x = self.actv(x)
                x = self.fct20(x)
                x = self.actv(x)
            else:
                print("Error: i == ", i, " is not supported now")
                sys.exit()
            if i == 0:
                x_out = x
            else:
                x_out = torch.cat((x_out, x), axis=1)
#                x_out *= x


            if i == self.nmff-1:
                x = self.outl(x_out)

        return x

    def pred_t(self, x, xs):
        ndata = len(x)
        xj = torch.empty((ndata,6))
        xjr = torch.empty((ndata,6))
        xj[:, 0] = x[:, 0]
        xj[:, 1] = x[:, 1]
        xj[:, 2] = x[:, 2]
        xj[:, 3] = xs[:, 0]
        xj[:, 4] = xs[:, 1]
        xj[:, 5] = xs[:, 2]
        xjr[:, 0] = xs[:, 0]
        xjr[:, 1] = xs[:, 1]
        xjr[:, 2] = xs[:, 2]
        xjr[:, 3] = x[:, 0]
        xjr[:, 4] = x[:, 1]
        xjr[:, 5] = x[:, 2]
        x = 0.5*self.pred_arctanhveff(xj)+0.5*self.pred_arctanhveff(xjr)
        x = torch.tanh(x)
        x = (x+1.)/2.*(self.vtotmax*1.1-self.vtotmin*0.9)+self.vtotmin*0.9
#        x = (x+1.)/2.*(self.vtotmax-self.vtotmin)+self.vtotmin
        x = 1./x
#        x = torch.tanh(x)
#        x = (x+1.)/2.*(self.taumax-self.taumin)+self.taumin
        return x

    def forward_pde_nohp(self, x, xs, v):
        ndata = len(x)
        tau = self.pred_t(x, xs)
        t0 = ((x[:,0]-xs[:,0])**2+(x[:,1]-xs[:,1])**2+(x[:,2]-xs[:,2])**2)**0.5
        T = tau[:,0]*t0
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        p = (T_x[:,0]**2+T_x[:,1]**2+T_x[:,2]**2)**(-0.5)-v
        p /= (len(v)**0.5)
        return p

    def forward_pde_nohp_nonormalization(self, x, xs, v):
        ndata = len(x)
        tau = self.pred_t(x, xs)
        t0 = ((x[:,0]-xs[:,0])**2+(x[:,1]-xs[:,1])**2+(x[:,2]-xs[:,2])**2)**0.5
        T = tau[:,0]*t0
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        p = (T_x[:,0]**2+T_x[:,1]**2+T_x[:,2]**2)**(-0.5)-v
        return p

            
    def traveltime_pred_obs(self, x, xs):
        ndata = len(x)
#        t0 = (((x[:, 0]-xs[:, 0])**2+(x[:, 1]-xs[:, 1])**2)**0.5)
        t0 = ((x[:,0]-xs[:,0])**2+(x[:,1]-xs[:,1])**2+(x[:,2]-xs[:,2])**2)**0.5
        tau = self.pred_t(x, xs)
        t = tau[:, 0]*t0
        return t
            
    def traveltime_pred_obs_reciprocity(self, x, xs):
        ndata = len(x)
        t0 = ((x[:,0]-xs[:,0])**2+(x[:,1]-xs[:,1])**2+(x[:,2]-xs[:,2])**2)**0.5
        tau = self.pred_t(xs, x)
        t = tau[:, 0]*t0
        return t

    def traveltime_pred_xzgrid(self, X, Z, y, xs):
        x = X.ravel()
        z = Z.ravel()
        yuni = torch.ones(len(x))*y
        x = torch.vstack([torch.vstack([x,yuni]),z]).T
        ndata = len(x)
        xs_list = torch.ones(ndata,3)
        xs_list[:, 0] *= xs[0]
        xs_list[:, 1] *= xs[1]
        xs_list[:, 2] *= xs[2]
        tau = self.pred_t(x,xs_list)
        t0 = ((x[:,0]-xs[0])**2+(x[:,1]-xs[1])**2+(x[:,2]-xs[2])**2)**0.5
        t = tau[:, 0]*t0
        t = (t).reshape(len(X), len(X[0, :])).detach()#.numpy().copy()
        return t

    def traveltime_pred_xygrid(self, X, Y, z, xs):
        x = X.ravel()
        y = Y.ravel()
        zuni = torch.ones(len(x))*z
        x = torch.vstack([torch.vstack([x,y]),zuni]).T
        ndata = len(x)
        xs_list = torch.ones(ndata,3)
        xs_list[:, 0] *= xs[0]
        xs_list[:, 1] *= xs[1]
        xs_list[:, 2] *= xs[2]
        tau = self.pred_t(x,xs_list)
        t0 = ((x[:,0]-xs[0])**2+(x[:,1]-xs[1])**2+(x[:,2]-xs[2])**2)**0.5
        t = tau[:, 0]*t0
        t = (t).reshape(len(X), len(X[0, :])).detach()#.numpy().copy()
        return t

    def loss_obs(self, x_obs, xs_obs, y_obs):
        Tobs = self.traveltime_pred_obs(x_obs, xs_obs)
        return self.loss_function(Tobs, y_obs)

    def loglikelihood(self, theta, x_b, y_b, stdobs_b, oriT):
        Tobs = self.traveltime_pred_obs(x_b, theta)
        return -0.5*((Tobs-(y_b-oriT))/stdobs_b)@((Tobs-(y_b-oriT))/stdobs_b)

    def RMSE(self, theta, x_b, y_b, oriT):
        Tobs = self.traveltime_pred_obs(x_b, theta)
        return ((Tobs-(y_b-oriT))@(Tobs-(y_b-oriT))/len(y_b))**0.5
    """
    def is_within_domain(self, coords):
        
        #Check if input coordinates are within the model domain defined by self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax.

        #Args:
        #    coords (torch.Tensor or np.ndarray): Coordinates to check, shape (..., 3)

        #Returns:
        #    torch.BoolTensor or np.ndarray: Boolean mask of shape (...,) indicating if each point is within the domain.
        
        # Collect domain bounds
        bounds_min = [self.xmin, self.ymin, self.zmin]
        bounds_max = [self.xmax, self.ymax, self.zmax]

        if isinstance(coords, np.ndarray):
            mask = np.all((coords >= bounds_min) & (coords <= bounds_max), axis=-1)
            return mask
        elif isinstance(coords, torch.Tensor):
            bounds_min_t = torch.tensor(bounds_min, dtype=coords.dtype, device=coords.device)
            bounds_max_t = torch.tensor(bounds_max, dtype=coords.dtype, device=coords.device)
            mask = ((coords >= bounds_min_t) & (coords <= bounds_max_t)).all(dim=-1)
            return mask
        else:
            raise TypeError("coords must be a torch.Tensor or np.ndarray")
    """
