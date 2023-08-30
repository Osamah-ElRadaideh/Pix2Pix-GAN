# Pix2Pix-GAN
Pix2Pix GAN implementation for image colourisation

training runs with sacred with 4 configurables, the batch size, discriminator learning rate, generator learning rate and evaluation steps.



to run a training with a batch size of 16, d_lr=1e-3, g_lr=1e-4, steps_per_eval=500, open the terminal and navigate to the folder directory and run  ```python -m train with batch_size=16, d_lr=1e-3 g_lr=1e-4 steps_per_eval=500```.

due to the limited hardware capabilities, i had to use a small image dataset with dimensions of 64x64.


ground truth images:
  <img src="/images/ground_truth.png" alt="gt" >
  
  <img src="/images/set1_GT.png" alt="set1_GT" >


greyscale images:

<img src="/images/bw.png" alt="bw" >
<img src="/images/set1_BW.png" alt="set1_bw" >


colourised images:

<img src="/images/colourised.png" alt="colourised" >
<img src="/images/set1_C.png" alt="set1_C" >












