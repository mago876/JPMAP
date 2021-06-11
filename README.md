# JPMAP

Implementation of:
[González, M., Almansa, A., & Tan, P. (2021). Solving Inverse Problems by Joint Posterior Maximization with Autoencoding Prior. arXiv preprint arXiv:2103.01648.](https://arxiv.org/abs/2103.01648)  

To install the needed conda environment, run `conda env create -f jpmap_env.yml`.  
Then activate the environment with `conda activate jpmap`.  

The bash script `run_experiments.sh` contains example code for training a VAE and running JPMAP for different inverse problems.  
The parameters for running JPMAP are described in `run_algorithms.py`.
