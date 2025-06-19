import time
import torch
import sys
import os
import argparse
import warnings
import HypoNet_Nankai.HypoNet.HypoNet as HypoNet
import HypoNet_Nankai.HypoNet_DeeperPINN.HypoNet_DeeperPINN as HypoNet_DeeperPINN
import HypoNet_Nankai.utils.hypo_posterior_PINN as hypo_posterior_PINN
import HypoNet_Nankai.utils.io as io
import HypoNet_Nankai.utils.hypocenter_initialization as hypocenter_initialization

warnings.filterwarnings("ignore")

def setup_pytorch_threading():
    """Configure PyTorch threading based on environment variables."""
    if 'OMP_NUM_THREADS' in os.environ:
        num_threads = int(os.environ['OMP_NUM_THREADS'])
        print(f"PyTorch using {num_threads} threads (from OMP_NUM_THREADS)")
    else:
        num_threads = 1
        print("PyTorch using 1 thread (from default)")

    torch.set_num_threads(num_threads)
    # Also set MKL threads for Intel MKL operations
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = str(num_threads)

def setup_data_type_and_device(BFP):
    """Setup data type and device configuration."""
    # Setting of default data type
    if BFP.dtype == "double":
        cuda_ttype = 'torch.cuda.DoubleTensor'
        cpu_ttype = 'torch.DoubleTensor'
        torch_dtype = torch.float64
    elif BFP.dtype == "single":
        cuda_ttype = 'torch.cuda.FloatTensor'
        cpu_ttype = 'torch.FloatTensor'
        torch_dtype = torch.float32
    else:
        print("Error: invalid data type")
        sys.exit()

    torch.set_default_dtype(torch_dtype)
    torch.set_default_tensor_type(cpu_ttype)
    
    if torch.cuda.is_available():
        if BFP.deviceid == 0:
            device2 = torch.device('cpu')
        else:
            torch.backends.cudnn.benchmark = True
            device2 = torch.device("cuda:0")
    else:
        device2 = torch.device('cpu')
    
    return device2, cuda_ttype, cpu_ttype, torch_dtype

def setup_models(device2, cuda_ttype, cpu_ttype):
    """Initialize HypoNet models."""
    if device2 != torch.device("cpu"):
        torch.set_default_tensor_type(cuda_ttype)  
        with torch.cuda.device(device2):
            mypinn_t = mymultipinn.MyMultiPINN("input", BFP.fact_name, BFP.nhlt, BFP.nnt, BFP.nmff, BFP.FFsigma)
            mypinn_t.input_models("input", device2)
        torch.set_default_tensor_type(cpu_ttype)
    else:
        hyponet_p = HypoNet.model(inputdir="./input_p/", device=0)
        hyponet_s = HypoNet_DeeperPINN.model(inputdir="./input_s/", device=0)
        return hyponet_p, hyponet_s

def process_single_event(i_s, eventdir, outputdir, hyponet_p, hyponet_s, BFP, device2, torch_dtype, dry_run):
    """Process a single event for hypocenter determination."""
    input_fn = f"./{eventdir}/event.{str(i_s).zfill(8)}.txt"
    output_fn = f"./{outputdir}/result.{str(i_s).zfill(8)}.txt"
    log_fn = f"./{outputdir}/log.{str(i_s).zfill(8)}.txt"
    
    with open(log_fn, "w") as log_file:
        log_file.write(f"Event ID: {i_s}\n")

        # Load observation data
        x_obs_p, y_obs_p, meas_err_p, x_obs_s, y_obs_s, meas_err_s, nobs_p, ns_p, iotype = io.input_data(input_fn, torch_dtype)
        log_file.write(f"IO type: {iotype}\n")

        Temperature = 1.0
        npastepoch = 0

        # Check if observations are within domain
        r_p_mask = hyponet_p.is_within_domain(x_obs_p[:,0], x_obs_p[:,1])
        
        # make observation of s-wave empty forcibly
        x_obs_s = torch.zeros((0, 3))
        y_obs_s = torch.zeros((0, 1))
        meas_err_s = torch.tensor([])
        r_s_mask = torch.tensor([])
        #r_s_mask = hyponet_s.is_within_domain(x_obs_s[:,0], x_obs_s[:,1])
    
        if not r_p_mask.all() or not r_s_mask.all():
            if not r_p_mask.all():
                log_file.write("Some elements of p-wave receivers are out of domain\n")
                for idx, mask in enumerate(r_p_mask):
                    if not mask:
                        x_p_i = x_obs_p[idx]
                        log_file.write(f"{x_p_i[0].item()}, {x_p_i[1].item()}, {x_p_i[2].item()}\n")
            if not r_s_mask.all():  
                log_file.write("Some elements of s-wave receivers are out of domain\n")
                for idx, mask in enumerate(r_s_mask):
                    if not mask:
                        x_s_i = x_obs_s[idx]
                        log_file.write(f"{x_s_i[0].item()}, {x_s_i[1].item()}, {x_s_i[2].item()}\n")
            
            log_file.write(f"Event ID: {i_s} is failed (dry run)\n")
            return False
        elif dry_run:
            log_file.write(f"Event ID: {i_s} is valid (dry run)\n")
            return True
        else:
            # Initialize hypocenter location
            xs, a, b = hypocenter_initialization.initialize_hypocenter_location_TT(x_obs_p, y_obs_p)
            log_file.write(f"Initial hypocenter location: {xs}\n")

            sigma_frac = 0.01
            hp = hypo_posterior_PINN.hypo_posterior_PINN(hyponet_p, hyponet_s, Temperature, sigma_frac, a, b, device2)

            # Mapping to the unconstrained space
            xs_const = hp.conversion(xs, device="cpu").clone().detach()
            optimizer = torch.optim.LBFGS([xs_const], tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn="strong_wolfe")

            # Prepare observation data
            y_obs_p_e = (y_obs_p.T)[0].reshape(1, len((y_obs_p.T)[0]))
            y_obs_s_e = (y_obs_s.T)[0].reshape(1, len((y_obs_s.T)[0]))

            # L-BFGS optimization loop
            start_time = time.time()
            for it in range(BFP.max_iter):
                def closure():
                    optimizer.zero_grad()
                    grad, loss = hp.get_grad(xs_const, x_obs_p, y_obs_p_e, x_obs_s, y_obs_s_e, meas_err_p, meas_err_s)
                    loss = -loss.clone().detach()
                    xs_const.grad = -grad.clone().detach()
                    return loss
                optimizer.step(closure)

                if it % int(round(BFP.max_iter/min(5,BFP.max_iter))) == 0 or it == BFP.max_iter-1:
                    grad, loss = hp.get_grad(xs_const, x_obs_p, y_obs_p_e, x_obs_s, y_obs_s_e, meas_err_p, meas_err_s)
                    loss = -loss.clone().detach()
                    log_file.write(f"Loss at Loop {it+npastepoch}: {loss.item()}\n")
                    xs = hp.inv_conversion(xs_const, device="cpu")

            elapsed = time.time() - start_time 
            log_file.write(f"L-BFGS for {i_s+1}th source took {elapsed}\n")

            # Mapping to the original space
            xs = hp.inv_conversion(xs_const, device="cpu")

            # Check if final solution is within domain
            s_enu_mask = hyponet_p.is_within_domain(xs[0], xs[1], xs[2])
            if not s_enu_mask.all():
                log_file.write("Source is out of domain\n")
                log_file.write(f"Coordinate: {xs[0]}, {xs[1]}, {xs[2]}\n")
                log_file.write(f"Event ID: {i_s} is failed\n")
                return False
                
            else:
                # Calculate posterior covariance matrix for estimation error
                xs.requires_grad = True
                xs_const = hp.conversion(xs, device="cpu")
                logp = hp.get_logp(xs_const, x_obs_p, y_obs_p_e, x_obs_s, y_obs_s_e, meas_err_p, meas_err_s)
                grad, = torch.autograd.grad(-logp, xs, retain_graph=True, create_graph=True)
                hessian = torch.empty((3,3))
                for i, g in enumerate(grad):
                    h, = torch.autograd.grad(g, xs, retain_graph=True)
                    hessian[i] = h                   
                cov = torch.linalg.inv(hessian)

                # Output results
                xs_np = xs.reshape(1,3).clone().detach().numpy()
                cov_np = cov.clone().detach().numpy()

                # Write to file
                io.output(output_fn, xs_np, cov_np, iotype)
                log_file.write(f"Event ID: {i_s} is succeeded\n")
                return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hypocenter location with PINN")
    parser.add_argument('--src_s', type=int, default=0, help='Start event index')
    parser.add_argument('--src_e', type=int, default=0, help='End event index')
    parser.add_argument('--eventdir', type=str, default='', help='Directory containing event files')
    parser.add_argument('--outputdir', type=str, default='', help='Directory to save output files')
    parser.add_argument('--dry_run', action='store_true', help='Run the script in dry run mode (hypocenter will not be determined)')
    args = parser.parse_args()

    src_s = args.src_s  
    src_e = args.src_e
    eventdir = args.eventdir
    outputdir = args.outputdir
    dry_run = args.dry_run

    print(f"Input: ./{eventdir}/event.{str(src_s).zfill(8)}.txt ... ./{eventdir}/event.{str(src_e).zfill(8)}.txt")
    print(f"Output: ./{outputdir}/*")

    # Setup PyTorch threading
    setup_pytorch_threading()
    
    # Load parameters
    BFP = io.param()
    BFP.read_dtype()
    BFP.read()

    # Setup data type and device
    device2, cuda_ttype, cpu_ttype, torch_dtype = setup_data_type_and_device(BFP)
    
    # Setup models
    hyponet_p, hyponet_s = setup_models(device2, cuda_ttype, cpu_ttype)

    # Process events
    for i_s in range(src_s, src_e+1):
        
        
        success = process_single_event(i_s, eventdir, outputdir, hyponet_p, hyponet_s, BFP, device2, torch_dtype, dry_run)
        if success:
            print(f"Event {i_s} processed successfully")
        else:
            print(f"Event {i_s} failed")
    
    print("Done")

if __name__=='__main__':
    main()
