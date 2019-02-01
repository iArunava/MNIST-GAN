# Simple Generative Adversarial Networks

In this repository, I create a simple generator and discriminator to generate new MNIST images. <br/>
The Generator and the Discriminator are both Linear models, a brief overview of how the models look are given further down 
the readme. In this repostitory, I generate mew images of the classic MNIST datasets, which include
- KMNIST
- MNIST
- Fashion MNIST


Pretrained models for all the datasets are made available in this repository. <br/>

Further, one can even train their own Generator and Discriminator using just one command with the help of this repository.

## How to use

0. Clone the repo and cd into it
```
git clone https://github.com/iArunava/MNIST-GAN.git
cd MNIST-GAN/
```

1. Start Training
```
python3 init.py --mode train --dataset kmnist
```

2. Predict using the saved pretrained models.
<small> Note: the saved models are treated as checkpoint files, which has the `state_dict` key. </small>
```
python3 init.py --mode predict --dataset kmnist
```
or
```
python3 init.py --mode predict -dpath /path/to/discriminator.pth -gpath /path/to/generator.pth
```

## How the model looks?

### Discriminator

```
Discriminator(
  (linear1): Linear(in_features=784, out_features=512, bias=True)
  (linear2): Linear(in_features=512, out_features=256, bias=True)
  (linear3): Linear(in_features=256, out_features=128, bias=True)
  (linear4): Linear(in_features=128, out_features=1, bias=True)
  (dropout): Dropout(p=0.3)
)
```

### Generator

```
Generator(
  (fc1): Linear(in_features=100, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=784, bias=True)
  (dropout): Dropout(p=0.3)
)
```

## Results using the pretrained models made available within this repository

### MNIST

![mnist gan](https://user-images.githubusercontent.com/26242097/52112342-34eeea00-262c-11e9-8aa5-b0937f7ba192.png)

### Fashion MNIST

![gan generated fashionmnist](https://user-images.githubusercontent.com/26242097/52112319-243e7400-262c-11e9-8f79-e2badb69a887.png)

### KMNIST

![kmnist simple gan](https://user-images.githubusercontent.com/26242097/52112635-13423280-262d-11e9-9a4c-4ae9d34e2e9a.png)

## License

The code in this repository is made available for free. Feel free to fork this repository and start playing with it.
