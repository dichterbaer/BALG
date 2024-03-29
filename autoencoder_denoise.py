import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)




def add_noise(inputs,noise_factor=0.3):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy


def add_blur(inputs, sigmas=(0.1, 2.0)):
    transform = T.GaussianBlur(kernel_size=(5, 5), sigma=sigmas)
    return transform(inputs)


def add_blur_and_noise(inputs, noise_factor= 0.3, sigmas=(0.1, 2.0)):
    transform = T.GaussianBlur(kernel_size=(5, 5), sigma=sigmas)
    blurry = transform(inputs)
    noisy_blurry = blurry+torch.randn_like(blurry) * noise_factor
    noisy_blurry = torch.clip(noisy_blurry,0.,1.)
    return noisy_blurry


### Training function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        # image_noisy = add_noise(image_batch,noise_factor)
        # image_noisy_blurry = add_blur(image_noisy)
        image_noisy_blurry = add_blur_and_noise(image_batch,noise_factor)
        image_batch = image_batch.to(device)
        image_noisy_blurry = image_noisy_blurry.to(device)    
        # Encode data
        encoded_data = encoder(image_noisy_blurry)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            # image_noisy = add_noise(image_batch,noise_factor)
            # image_noisy_blurry = add_blur(image_noisy)
            image_noisy_blurry = add_blur_and_noise(image_batch,noise_factor)
            image_noisy_blurry = image_noisy_blurry.to(device)
            # Encode data
            encoded_data = encoder(image_noisy_blurry)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs_den(encoder,decoder,n=10,noise_factor=0.3, additional_info=""):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}    
    for i in range(n):

      ax = plt.subplot(3,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0)
    #   image_noisy = add_noise(img,noise_factor) 
    #   image_noisy_blurry = add_blur(image_noisy)    
      image_noisy_blurry = add_blur_and_noise(img,noise_factor)
      image_noisy_blurry = image_noisy_blurry.to(device)

      encoder.eval()
      decoder.eval()

      with torch.no_grad():
         rec_img  = decoder(encoder(image_noisy_blurry))

      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(3, n, i + 1 + n)
      plt.imshow(image_noisy_blurry.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Corrupted images (Blurry + Noise)')

      ax = plt.subplot(3, n, i + 1 + n + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)     
    # plt.show()
    plt.savefig('results/result_images/deblurring_denoising_ae'+additional_info+'.png')
    # plt.close()

bottleneck_size = [2, 4, 8, 16, 32, 64, 128]
### Initialize the two networks
for d in bottleneck_size:

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)


        ### Training cycle
    noise_factor = 0.3
    num_epochs = 30
    history_da={'train_loss':[],'val_loss':[]}

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training (use the training function)
        train_loss=train_epoch_den(
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optim,noise_factor=noise_factor)
        ### Validation  (use the testing function)
        val_loss = test_epoch_den(
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=valid_loader, 
            loss_fn=loss_fn,noise_factor=noise_factor)
        # Print Validationloss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        additional_info = f"_epoch_{epoch}_val_loss_{val_loss:.3f}_bottleneck_{d}_noise_factor_{noise_factor}" 
        plot_ae_outputs_den(encoder,decoder,noise_factor=noise_factor, additional_info = additional_info)

    #save the model
    torch.save(encoder.state_dict(), f'results/nets/encoder_denoise_deblur_first_blur_then_noise_bottleneck_{d}.pth')
    torch.save(decoder.state_dict(), f'results/nets/decoder_denoise_deblur_first_blur_then_noise_bottleneck_{d}.pth')


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

encoder.eval()
decoder.eval()

# with torch.no_grad():
#     # calculate mean and std of latent code, generated takining in test images as inputs 
#     images, labels = iter(test_loader).next()
#     images = images.to(device)
#     latent = encoder(images)
#     latent = latent.cpu()

#     mean = latent.mean(dim=0)
#     print(mean)
#     std = (latent - mean).pow(2).mean(dim=0).sqrt()
#     print(std)

#     # sample latent vectors from the normal distribution
#     latent = torch.randn(128, d)*std + mean

#     # reconstruct images from the random latent vectors
#     latent = latent.to(device)
#     img_recon = decoder(latent)
#     img_recon = img_recon.cpu()

#     fig, ax = plt.subplots(figsize=(20, 8.5))
#     show_image(torchvision.utils.make_grid(img_recon[:100],10,5))
#     plt.show()