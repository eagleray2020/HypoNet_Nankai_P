# Example script for running HypoNet Nankai for the numerical experiment presented in Agata et al. (2025)
# To run this script, you need to install "pygmt" and "geopy" python packages in addition to HypoNet Nankai

mkdir -p output

hyponetn_run --eventdir events --src_s 0 --src_e 47 --outputdir output

python plot_paper.py --x_s 0 --x_e 47

