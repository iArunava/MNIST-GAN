import torch
import numpy as np
from viz import *

def test(FLAGS):

    sample_size = FLAGS.sample_size
    z_size = FLAGS.z_size
    cuda = FLAGS.cuda
    g_path = FLAGS.g_path
    d_path = FLAGS.d_path

    # Load the models
    D = torch.load(d_path)
    G = torch.load(g_path)

    # Define some latent vectors
    z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    z = torch.from_numpy(z).float()

    if cuda:
        z = z.cuda()

    # Eval mode
    G.eval()

    rand_images = G(z)

    view_samples(0, [rand_images])
