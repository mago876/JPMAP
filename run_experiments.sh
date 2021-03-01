# Train VAEfc on MNIST (the following line generates the pretrained model included in this repository)
#python train_VAE.py --dataset mnist --network-structure VAEfc --latent-dim 8 --dvae-sigma 15 --lr 1e-4 --batch-size 128 --epochs 200 --exp-name VAEfc_MNIST_zdim8_dvae15_ELU


# Some experiments comparing CSGM and JPMAP (add "--save-iters" option to store JPMAP iterations)

# Inpainting (random mask)
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem missingpixels --sigma 10 --missing 0.80 --n-samples 10 --iters-jpmap 300 --uzawa --alpha 1 --exp-name missing_perc80_alpha1 --save-iters # Exponential multiplier method
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem missingpixels --sigma 10 --missing 0.80 --n-samples 10 --iters-jpmap 300 --exp-name missing_perc80_mapxz  # beta=1/gamma^2

# Denoising
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem denoising --sigma 110 --n-samples 10 --iters-jpmap 300 --uzawa --alpha 1 --exp-name denoising_sigma110_alpha1  # Exponential multiplier method
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem denoising --sigma 110 --n-samples 10 --iters-jpmap 300 --exp-name denoising_sigma110_mapxz  # beta=1/gamma^2

# Compressed Sensing
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem compsensing --sigma 10 --output-dim 140 --n-samples 10 --iters-jpmap 300 --uzawa --alpha 1 --exp-name compsensing_dim140_alpha1  # Exponential multiplier method
python run_algorithms.py --dataset mnist --model VAEfc_MNIST_zdim8_dvae15_ELU --problem compsensing --sigma 10 --output-dim 140 --n-samples 10 --iters-jpmap 300 --exp-name compsensing_dim140_mapxz  # beta=1/gamma^2
