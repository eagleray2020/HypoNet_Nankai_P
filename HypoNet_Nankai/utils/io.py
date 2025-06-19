import numpy as np
import torch 
import os
class param:
    def read_dtype(self):
        """
        with open("input/setting") as f:
            for i in range(16):
                s = f.readline()
            items = s.split()
            self.dtype = items[0]
        """
        self.dtype = "single"
    def read(self):

        self.max_iter = 5 # for L-BFGS
        self.deviceid = 0 # default is CPU
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(BASE_DIR+"/input/setting") as f:
            s = f.readline()
            s = f.readline()
            items = s.split()
            self.nhlt = int(items[0])
            self.nnt = int(items[1])
            self.fact_name = items[2]
            self.nmff = len(items)-3
            self.FFsigma = []
            for i in range(self.nmff):
                self.FFsigma.append(float(items[i+3]))
            """
            s = f.readline()
            s = f.readline()
            items = s.split()
            self.npart = int(items[0])

            s = f.readline()
            s = f.readline()

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.max_iter = int(items[0])
            self.thres = float(items[1])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.optimizer_name = items[0]
            self.lr = float(items[1])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.std_obs = float(items[0])

            s = f.readline()
            s = f.readline()
            items = s.split()
            self.deviceid = int(items[0])

            s = f.readline()
            s = f.readline()
            s = f.readline()
            s = f.readline()
            items = s.split()
            self.src_s = int(items[0])
            self.src_e = int(items[1])
            """
def input_data(input_fn, torch_dtype):

    parr, meas_err_p, sarr, meas_err_s, lat, lon, height, pcorr, scorr, iotype = parse_data(input_fn, torch_dtype)
    nsta = parr.shape[0]


    x_obs_p = torch.vstack((lon, lat, height*1e-3)).T
    x_obs_s = x_obs_p.clone()
    y_obs_p = (parr+pcorr).reshape(nsta,1)
    y_obs_s = (sarr+scorr).reshape(nsta,1)
    #meas_err_p = numerisor[:,1] 
    #meas_err_s = numeric_tensor[:,3] 

    nan_mask = torch.isnan(y_obs_p).squeeze()  # Remove extra dimensions
    valid_indices = ~nan_mask  # Get indices where y_obs_p does not have NaN
    # Remove rows with NaN
    
    y_obs_p = y_obs_p[valid_indices]
    x_obs_p = x_obs_p[valid_indices]
    meas_err_p = meas_err_p[valid_indices]

    nan_mask = torch.isnan(y_obs_s).squeeze()  # Remove extra dimensions
    valid_indices = ~nan_mask  # Get indices where y_obs_p does not have NaN
    # Remove rows with NaN
    y_obs_s = y_obs_s[valid_indices]
    x_obs_s = x_obs_s[valid_indices]
    meas_err_s = meas_err_s[valid_indices]

    nobs_p = len(y_obs_p)
    ns_p = len(y_obs_p[0,:])

    return x_obs_p, y_obs_p, meas_err_p, x_obs_s, y_obs_s, meas_err_s, nobs_p, ns_p, iotype

def parse_data(filename, torch_dtype):


    numeric_ndarray = np.genfromtxt(filename, skip_header=1)
    if numeric_ndarray.shape[1] == 9:
            #print(numeric_ndarray)
        numeric_tensor = torch.from_numpy(numeric_ndarray).to(torch_dtype)
        
        #print(numeric_tensor)
        return numeric_tensor[:,1], numeric_tensor[:,2], \
            numeric_tensor[:,3], numeric_tensor[:,4], \
            numeric_tensor[:,5], numeric_tensor[:,6], torch.zeros_like(numeric_tensor[:,6]), \
            numeric_tensor[:,7], numeric_tensor[:,8], "numpy"
    else:


        text_data = []
        numeric_data = []


        def safe_float(value):
            try:
                return float(value)
            except ValueError:
                return float('nan')

        with open(filename, 'r') as file:
            next(file)
            for line in file:
                parts = line.split()
                text_data.append(parts[:2])
                
                # Extract the numeric data part
                numbers = [safe_float(part) for part in parts[2:]]
                
                # Add zeros if 12th and 13th columns are missing
                while len(numbers) < 11:
                    numbers.append(0.0)
                
                numeric_data.append(numbers)

        text_list = [tuple(t) for t in text_data]
        numeric_array = np.array(numeric_data)
        numeric_tensor = torch.tensor(numeric_array, dtype=torch_dtype)
        

        return numeric_tensor[:,0], numeric_tensor[:,1], \
            numeric_tensor[:,2], numeric_tensor[:,3], \
            numeric_tensor[:,6], numeric_tensor[:,7], numeric_tensor[:,8], \
            numeric_tensor[:,9], numeric_tensor[:,10], "hypomh"

def output(output_fn, xs_np, cov_np, iotype):
        
        if iotype == "hypomh":
            
            # Prepare the data for each row
            row1 = f"999 999 999 999 999 999 {xs_np[0,1]} {xs_np[0,0]} {-xs_np[0,2]} 999"
            row2 = f"dummy 0. {np.sqrt(cov_np[1, 1]):.6f} {np.sqrt(cov_np[0, 0]):.6f} {np.sqrt(cov_np[2, 2]):.6f}"
            row3 = f"{cov_np[0, 0]} {cov_np[0, 1]} {cov_np[0, 2]} {cov_np[1, 1]} {cov_np[1, 2]} {cov_np[2, 2]}"

            with open(output_fn, 'w') as f:
                f.write(row1 + '\n')
                f.write(row2 + '\n')
                f.write(row3 + '\n')
        else: # numpy
            header1 = "#lon lat depth lon_err lat_err depth_err"
            header2 = "#Covariance matrix: xx xy xz yy yz zz"
            row1 = f"{xs_np[0,0]} {xs_np[0,1]} {-xs_np[0,2]} {np.sqrt(cov_np[0, 0]):.6f} {np.sqrt(cov_np[1, 1]):.6f} {np.sqrt(cov_np[2, 2]):.6f}"
            row2 = f"{cov_np[0, 0]} {cov_np[0, 1]} {cov_np[0, 2]} {cov_np[1, 1]} {cov_np[1, 2]} {cov_np[2, 2]}"
            with open(output_fn, 'w') as f:
                f.write(header1 + '\n')
                f.write(header2 + '\n')
                f.write(row1 + '\n')
                f.write(row2 + '\n')
                
                