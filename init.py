import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-bs', '--batch-size',
            type=int,
            default=64,
            help='The batch size')

    parser.add_argument('-dlr',
            type=float,
            default=0.002,
            help='The learning rate for the discriminator')

    parser.add_argument('-glr',
            type=float,
            default=0.002,
            help='The learning rate for the generator')

    parser.add_argument('-nw', '--num-workers',
            type=int,
            default=0,
            help='The batch size')

    parser.add_argument('-din',
            type=int,
            default=784,
            help='The number of input neurons to the discriminator')

    parser.add_argument('-dhid',
            type=int,
            default=128,
            help='The number of input neurons to the discriminator')

    parser.add_argument('-dout',
            type=int,
            default=1,
            help='The output, real/fake from discriminator \
                    No need to change this.')

    parser.add_argument('-zsize',
            type=int,
            default=100,
            help='The number of input neurons to the generator. The latent space')

    parser.add_argument('-gout',
            type=int,
            default=784,
            help='The number of output neurons from the generator. \
                    No need to change this as long you are working with MNIST \
                            Images.')

    parser.add_argument('-ghid',
            type=int,
            default=32,
            help='The number of hidden neurons in the generator')

    parser.add_argument('-c', '--cuda',
            type=bool,
            default=False,
            help='Whether to use GPU or not')

    parser.add_argument('-e', '--epochs',
            type=int,
            default=200,
            help='The number of epochs')

    parser.add_argument('-pe', '--p-every',
            type=int,
            default=50,
            help='To print loss and other stats at an interval of x epochs')

    parser.add_argument('-es', '--e-size',
            type=int,
            default=16,
            help='The sample size for the evaluation.')

    parser.add_argument('-ss', '--save-samples',
            type=bool,
            default=True,
            help='Whether to save samples or not')

    parser.add_argument('-pl', '--plot-losses',
            type=bool,
            default=True,
            help='Whether to plot losses or not')

    parser.add_argument('--mode',
            type=str,
            default='test',
            help='The mode whether to train or test')

    parser.add_argument('-dpath',
            type=str,
            default='./pretrained_models/KMNIST/model-d-kmnist.pt')

    parser.add_argument('-gpath',
            type=str,
            default='./pretrained_models/KMNIST/model-g-kmnist.pt')

    FLAGS, unparsed = parser.parse_known_args()

    # Check if cuda is available
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if mode == 'train':
        train(FLAGS)
    elif mode == 'test':
        test(FLAGS)
    else:
        raise RuntimeError('Invalid value passed for mode. \
                Valid arguments are: "train" and "test"')
