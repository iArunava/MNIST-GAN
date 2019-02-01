import torch
import numpy as np
from Discriminator import Discriminator
from Generator import Generator
from viz import *
import torch.nn.functional as F
import pickle as pkl

def test(FLAGS):

    sample_size = FLAGS.eval_size
    z_size = FLAGS.zsize
    cuda = FLAGS.cuda
    g_path = FLAGS.gpath
    d_path = FLAGS.dpath
    map_location = 'cuda' if cuda else 'cpu'

    # Load the models
    dckpt = torch.load(d_path, map_location=map_location)
    gckpt = torch.load(g_path, map_location=map_location)

    D = Discriminator(784, 128, 1)
    G = Generator(100, 32, 784)

    D.load_state_dict(dckpt['state_dict'])
    G.load_state_dict(gckpt['state_dict'])

    # Define some latent vectors
    z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    z = torch.from_numpy(z).float()

    if cuda:
        z = z.cuda()

    # Eval mode
    G.eval()

    rand_images = G(z)

    view_samples(0, [rand_images])
