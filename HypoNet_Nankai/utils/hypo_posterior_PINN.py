import numpy as np
import random
import torch
import time

class hypo_posterior_PINN():

    def __init__(self, hyponet_p, hyponet_s, Temperature, sigma_frac, a, b, device2):
        self.hyponet_p = hyponet_p
        self.hyponet_s = hyponet_s
        self.Temperature = Temperature
        self.device2 = device2
        self.t0 = 0.
        self.t = 0.
        self.sigma_frac = sigma_frac
        self.a_cpu = a.clone().detach()
        self.b_cpu = b.clone().detach()
        self.a = a.to(device2).clone().detach()
        self.b = b.to(device2).clone().detach()
        self.conversion = self.conversion_to_unconstrained_space
        self.inv_conversion = self.inv_conversion_to_unconstrained_space

    def get_grad(self, theta, x_obs_p, tt_obs_p, x_obs_s, tt_obs_s, meas_err_p, meas_err_s):
        device2 = self.device2

        theta_2 = theta.to(device2).clone().detach()
        theta_2.requires_grad = True
        inv_theta = self.inv_conversion(theta_2)

        x_obs_p_2 = x_obs_p.to(device2)
        inv_theta_p_xx, xobs_p_xx = torch.meshgrid(inv_theta[0], x_obs_p_2[:,0])
        inv_theta_p_yy, xobs_p_yy = torch.meshgrid(inv_theta[1], x_obs_p_2[:,1])
        inv_theta_p_zz, xobs_p_zz = torch.meshgrid(inv_theta[2], x_obs_p_2[:,2])
        tt_obs_p_2  = tt_obs_p.to(device2)

        x_obs_s_2 = x_obs_s.to(device2)
        inv_theta_s_xx, xobs_s_xx = torch.meshgrid(inv_theta[0], x_obs_s_2[:,0])
        inv_theta_s_yy, xobs_s_yy = torch.meshgrid(inv_theta[1], x_obs_s_2[:,1])
        inv_theta_s_zz, xobs_s_zz = torch.meshgrid(inv_theta[2], x_obs_s_2[:,2])
        tt_obs_s_2  = tt_obs_s.to(device2)

        start_time = time.time()

        if device2 != torch.device("cpu"):
            torch.set_default_tensor_type(cuda_ttype)
            with torch.cuda.device(device2):
                start_time = time.time()
                tt_p, errcodelist = self.hyponet_p.inference_torch(inv_theta_p_xx, inv_theta_p_yy, inv_theta_p_zz, xobs_p_xx, xobs_p_yy)
                tt_p = tt_p.reshape(tt_obs_p.shape[0], tt_obs_p.shape[1])
                tt_s, errcodelist = self.hyponet_s.inference_torch(inv_theta_s_xx, inv_theta_s_yy, inv_theta_s_zz, xobs_s_xx, xobs_s_yy)
                tt_s = tt_s.reshape(tt_obs_s.shape[0], tt_obs_s.shape[1])
                self.t0 += time.time()-start_time
                start_time = time.time()
                loglikelihood = self.loglikelihood_ryberg_ps_2(tt_p, tt_obs_p_2, tt_s, tt_obs_s_2, meas_err_p, meas_err_s)
                logprior = self.logprior_uniform(theta_2)
                grad, = torch.autograd.grad(torch.sum(loglikelihood)+torch.sum(logprior), theta_2)
                torch.cuda.current_stream().synchronize()
                self.t += time.time()-start_time
            torch.set_default_tensor_type(cpu_ttype)
            grad = grad.to(torch.device("cpu"))
            loglikelihood = loglikelihood.to(torch.device("cpu"))
        else:
            start_time = time.time()
            tt_p, errcodelist = self.hyponet_p.inference_torch(inv_theta_p_xx, inv_theta_p_yy, inv_theta_p_zz, xobs_p_xx, xobs_p_yy)
            tt_p = tt_p.reshape(tt_obs_p.shape[0], tt_obs_p.shape[1])
            tt_s, errcodelist = self.hyponet_s.inference_torch(inv_theta_s_xx, inv_theta_s_yy, inv_theta_s_zz, xobs_s_xx, xobs_s_yy)
            tt_s = tt_s.reshape(tt_obs_s.shape[0], tt_obs_s.shape[1])
            self.t0 += time.time()-start_time
            start_time = time.time()
            loglikelihood = self.loglikelihood_ryberg_ps_2(tt_p, tt_obs_p_2, tt_s, tt_obs_s_2, meas_err_p, meas_err_s)
            logprior = self.logprior_uniform(theta_2)
            grad, = torch.autograd.grad(torch.sum(loglikelihood)+torch.sum(logprior), theta_2)

            self.t += time.time()-start_time
        return grad/self.Temperature, loglikelihood/self.Temperature



    def get_logp(self, theta_2, x_obs_p_2, tt_obs_p_2, x_obs_s_2, tt_obs_s_2, meas_err_p, meas_err_s):
        inv_theta = self.inv_conversion(theta_2)

        inv_theta_p_xx, xobs_p_xx = torch.meshgrid(inv_theta[0], x_obs_p_2[:,0])
        inv_theta_p_yy, xobs_p_yy = torch.meshgrid(inv_theta[1], x_obs_p_2[:,1])
        inv_theta_p_zz, xobs_p_zz = torch.meshgrid(inv_theta[2], x_obs_p_2[:,2])

        inv_theta_s_xx, xobs_s_xx = torch.meshgrid(inv_theta[0], x_obs_s_2[:,0])
        inv_theta_s_yy, xobs_s_yy = torch.meshgrid(inv_theta[1], x_obs_s_2[:,1])
        inv_theta_s_zz, xobs_s_zz = torch.meshgrid(inv_theta[2], x_obs_s_2[:,2])

        start_time = time.time() 
        tt_p, errcodelist = self.hyponet_p.inference_torch(inv_theta_p_xx, inv_theta_p_yy, inv_theta_p_zz, xobs_p_xx, xobs_p_yy)
        tt_p = tt_p.reshape(tt_obs_p_2.shape[0], tt_obs_p_2.shape[1])
        tt_s, errcodelist = self.hyponet_s.inference_torch(inv_theta_s_xx, inv_theta_s_yy, inv_theta_s_zz, xobs_s_xx, xobs_s_yy)
        tt_s = tt_s.reshape(tt_obs_s_2.shape[0], tt_obs_s_2.shape[1])
        self.t0 += time.time()-start_time
        start_time = time.time()
        loglikelihood = self.loglikelihood_ryberg_ps_2(tt_p, tt_obs_p_2, tt_s, tt_obs_s_2, meas_err_p, meas_err_s)
        logprior = self.logprior_uniform(theta_2)
        self.t += time.time()-start_time
        return (torch.sum(loglikelihood)+torch.sum(logprior))/self.Temperature

    def calc_sigma(self, tt, meas_err):
        # See Hirata+1987PEPI
        sigma = ((meas_err**2+tt*self.sigma_frac**2)**0.5).clone().detach()
        return sigma


    def calc_sigma_hypoSVI(self, tt, sigmas):
        # See Smith2022GJI
        sigma_frac = sigmas[0]
        sigma_min  = sigmas[1]
        sigma_max  = sigmas[2]

        sigma = torch.where(tt*sigma_frac<sigma_min, sigma_min, tt*sigma_frac).clone().detach()
        sigma = torch.where(sigma>sigma_max, sigma_max, sigma)
        return sigma


    def loglikelihood_ryberg_ps_2(self, tt_p, tt_obs_p, tt_s, tt_obs_s, meas_err_p, meas_err_s):
        tt_obs_p_ravel = tt_obs_p.ravel()
        tt_obs_s_ravel = tt_obs_s.ravel()
        tt_p_ravel = tt_p.ravel()
        tt_s_ravel = tt_s.ravel()
        sigma_p = np.array([0.02, 0.1, 1])
        sigma_s = np.array([0.04, 0.1, 1])
        std_obs_p = np.sqrt(self.calc_sigma_hypoSVI(tt_obs_p_ravel, sigma_p)**2.+meas_err_p**2.)
        std_obs_s = np.sqrt(self.calc_sigma_hypoSVI(tt_obs_s_ravel, sigma_s)**2.+meas_err_s**2.)
        tavg_p = torch.mean(tt_p_ravel-tt_obs_p_ravel)
        tavg_s = torch.mean(tt_s_ravel-tt_obs_s_ravel)
        op = -0.5*( ((tt_p_ravel-tt_obs_p_ravel-tavg_p)/std_obs_p)@((tt_p_ravel-tt_obs_p_ravel-tavg_p)/std_obs_p) 
                  + ((tt_s_ravel-tt_obs_s_ravel-tavg_s)/std_obs_s)@((tt_s_ravel-tt_obs_s_ravel-tavg_s)/std_obs_s) )
        return op

    def logprior_uniform(self, theta):
        op = torch.sum(torch.log(torch.exp(-theta)/((1+torch.exp(-theta))**2)))
        return op

    def conversion_to_unconstrained_space(self, param_in, device="gpu"):
        if device=="cpu":
            a = self.a_cpu
            b = self.b_cpu
        else:
            a = self.a
            b = self.b

        param_out = torch.log(param_in-a)-torch.log(b-param_in)

        return param_out

    def inv_conversion_to_unconstrained_space(self, param_in, device="gpu"):
        if device=="cpu":
            a = self.a_cpu
            b = self.b_cpu
        else:
            a = self.a
            b = self.b
        param_out = a+(b-a)/(1.+torch.exp(-param_in))
        return param_out