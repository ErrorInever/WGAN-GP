## WGAN-GP
PyTorch implementation of WGAN with gradient penalty introduced in the paper: [WGAN](https://arxiv.org/abs/1511.06434), [WGAN-GP](https://arxiv.org/abs/1704.00028)

## Hyperparameters
All hyperparameters of this implementation specified in config file config.py

## Dataset
Original Dataset: https://www.kaggle.com/soumikrakshit/anime-faces
![dataset](https://raw.githubusercontent.com/ErrorInever/DCGAN-Anime-Faces/master/data/image_demonstration/Figure_1.png)

## Results
Training time 1h 30m 48s (GPU Tesla P100-PCIE-16GB) (10 epochs)
![loss_gen](https://raw.githubusercontent.com/ErrorInever/WGAN-GP/main/data/image_demonstration/gen_loss.png)


![fixed_noise](https://raw.githubusercontent.com/ErrorInever/WGAN-GP/main/data/image_demonstration/epoch(10)-batch(980).jpg)


Interpolation
![int](https://raw.githubusercontent.com/ErrorInever/WGAN-GP/main/data/image_demonstration/__results___21_1.png)

## ARGS and runs
    optional arguments:
      --data_path            path to dataset folder
      --seed                 seed value, default=7889
      --checkpoint_path      path to checkpoint.pth.tar
      --out_path             path to output folder
      --resume_id            wandb id of project for resume metric
      --device               use device, can be - cpu, cuda, tpu, if not specified: use gpu if available

      Other paths and other parameters you can set up in config.py
   > for example: python3 train.py --data_path 'anime_dataset'
    
   
## Inference
    optional arguments:
      --path_ckpt            Path to checkpoint of model
      --num_samples          Number of samples
      --steps                Number of step interpolation
      --device               cpu or gpu
      --out_path             Path to output folder, default=save to project folder
      --gif                  reate gif
      --grid                 Draw grid of images
      --z_size               The size of latent space, default=128
      --img_size             Size of output image
      --resize               if you want to resize images
