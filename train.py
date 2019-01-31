import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from .Discriminator import Discriminator
from .Generator import Generator
from viz import *

def train(FLAGS):
    
    # Define the hyperparameters
    batch_size = FLAGS.batch_size
    d_lr = FLAGS.d_lr
    g_lr = FLAGS.g_lr
    num_workers = FLAGS.num_workers
    d_in = FLAGS.d_in
    d_hid = FLAGS.d_hid
    d_out = FLAGS.d_out
    z_size = FLAGS.z_size
    g_out = FLAGS.g_out
    g_hid = FLAGS.g_hid
    cuda = FLAGS.cuda
    epochs = FLAGS.epochs
    p_every = FLAGS.p_every
    e_size = FLAGS.eval_size
    save_samples = FLAGS.save_samples

    # Define the transforms
    transform = transforms.ToTensor()

    # Get the training datasets
    train_data = datasets.MNIST(root='data', train=True,
                                download=True, transforms=transform)
    
    # Prepare the DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers)

    
    # Instantiate the network
    D = Discriminator(d_in, d_hid, d_out)
    G = Generator(g_in, g_hid, g_out)

    # Create optimizers
    d_opt = optim.Adam(D.parameters(), d_lr)
    g_opt = optim.Adam(G.parameters(). g_lr)

    # Get some fixed data for evaluating.
    fixed_z = np.random.uniform(-1, 1, size=(e_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    
    # Move to cuda if available
    if cuda:
        fixed_z = fixed_z.cuda()
        D.cuda()
        G.cuda()

    
    # Train the network
    print ('[INFO]Starting Training...')

    D.train()
    G.train()

    for epoch in range(epochs):
        
        for batch_i, (real_images, _) in enumerate(train_loader):
            
            if cuda:
                real_images = real_images# Train the network
                print ('[INFO]Starting Training...')

                D.train()
                G.train()

                for epoch in range(epochs):
                    
                    for batch_i, (real_images, _) in enumerate(train_loader):
                        
                        if cuda:
                            real_images = real_images.cuda()

                        batch_size = real_images.size(0)

                        # Rescaling the images
                        real_images = real_images*2 - 1

                        ### Train the Discriminator ###
                        d_opt.zero_grad()
                        
                        # Get the output of the discriminator using real_images
                        d_out = D(real_images)
                        
                        # Get the loss of the discriminator for the real_images
                        r_loss = real_loss(d_out, smooth=True)
                        
                        # Generate some latent samples to pass to the generator
                        z = np.random.uniform(-1, 1, size(batch_size, z_size))
                        z = torch.from_numpy(z).float()
                        if cuda:
                            z = z.cuda()

                        # Get fake images using the created latent samples
                        fake_images = G(z)

                        # Compute the discriminator losses on fake images
                        d_fake = D(fake_images)
                        f_loss = fake_loss(d_fake)

                        # add up real and fake losses and perform backprop
                        d_loss = r_loss + f_loss
                        d_loss.backward()
                        d_opt.step()

                        ### Train the Generator ###
                        g_opt.zero_grad()

                        # Train with fake images and flipped labels
                        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                        z = torch.from_numpy(z).float()

                        if cuda:
                            z = z.cuda()

                        fake_images = G(z)

                        # Compute discriminator losses on fake images
                        # using flipped labels
                        d_fake = D(fake_images)

                        # perform backprop
                        g_loss = real_loss(d_fake)
                        g_loss.backward()
                        g_optimizer.step()

                        # Print some stats
                        if batch_i % p_every == 0:
                            print ('Epoch [{:5d} / {:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.\
                                format(epoch+1, epochs, d_loss.item(), g_loss.item()))

                    # Append discriminator loss and generator loss
                    losses.append((d_loss.item(), g_loss.item()))

                    # Evaluate the model
                    G.eval()
                    samples_z = G(fixed_z)
                    samples.append(samples_z)
                    G.train()

                if save_samples:
                    with open('train_samples.pkl', 'wb') as f:
                        pkl.dump(samples, f)
