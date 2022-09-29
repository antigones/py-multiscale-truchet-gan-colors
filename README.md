# py-multiscale-truchet-gan-colors

Training a GAN to generate Multiscale Truchet with Colors

## Installing requirements in virtual environment

After creating/activating the virtual environment:

```shell
pip install -r requirements.txt
```

Create the following directories:

- ./imgs/train (for training set images)
- ./logs (for color GAN training logs)
- ./output (for color GAN generated outputs)
- ./output-gan (for color GAN training process images)

For conditional GAN:

- ./cond_gan_output (for conditional color GAN generated outputs)
## Generating training set

The training set is generated using the script augment_dataset.py:

```shell
python augment_dataset.py
```

Training images are generated in folder "imgs/train".

## Train the GAN

To train the GAN, lauch:

```shell
python tf_ms_truchet_gan_color.py
```

### Accessing Tensorboard

Launch Tensorboard server and follow on-screen instructions:

```shell
tensorboard --logdir logs
```

### Using saved model weights

If weights are already available, comment the following two lines in tf_truchet_gan.py:

```shell
train(train_dataset, EPOCHS)
make_animation()
```

and launch the script again to generate a new set of images:

```shell
python tf_ms_truchet_gan_color.py
```

# Conditional GAN

Launch:

```shell
python conditional_gan.py
```

Model will be saved in script directory, images in output will be saved to /cond_gan_output.