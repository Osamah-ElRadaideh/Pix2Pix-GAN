import numpy as np
import torch
import torch.nn as nn
from models import Generator, Discriminator, gen_loss, disc_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import AHE, collate
import lazy_dataset
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sw = SummaryWriter()
#load the files
def load_img(example):
    bw = cv2.imread(example['image_path'], 0)
    img = cv2.imread(example['image_path'])
    example['image'] = bw.astype(np.float32) / 255.0
    example['target'] = img.astype(np.float32) / 255.0
    return example

def prepare_dataset(dataset,batch_size=16):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size, drop_last=True)
    dataset = dataset.map(collate)
    return dataset

path = 'ckpt_latest.pth'
def main():
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5

    db = AHE()
    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    steps = 0
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.train()
    disc.train()
    g_optim = torch.optim.Adam(gen.parameters())
    d_optim = torch.optim.Adam(disc.parameters())
    aux_criterion = nn.L1Loss()
    for epoch in range(10000):
        epoch_g_loss = 0
        epoch_d_loss = 0
        train_ds = prepare_dataset(t_ds)
        valid_ds = prepare_dataset(v_ds,batch_size=16)
        for index,batch in enumerate(tqdm(train_ds)):
            g_optim.zero_grad()
            d_optim.zero_grad()
            images = batch['target']
            bws = batch['image']

            # cv2 loads images as (h,w,3), models take(3,h,w)
            images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2) 
            bws = torch.tensor(np.array(bws)).to(device)
            fakes = gen(bws)

            #*********************  
            
            # discriminator step

            #*********************

            d_real = disc(images)
            d_fake = disc(fakes.detach())
            loss_d = disc_loss(d_real, d_fake) * 0.5
            loss_d.backward()
            d_optim.step()
            epoch_d_loss += loss_d.item()


            #******************* 
            # generator step
        
            #*******************
            fakeouts = disc(fakes)
            loss_g = gen_loss(fakeouts) + aux_criterion(fakes, images) * 10 
            loss_g.backward()
            g_optim.step()
            epoch_g_loss += loss_g.item()
        
           
            if steps % 1000 == 0:
                gen.eval()
                disc.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_ds[0:1]):
                        images = batch['target']
                        bws = batch['image']
                        images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2)
                        bws = torch.tensor(np.array(bws)).to(device)
                        generated = gen(bws)
                        d_fake = disc(fakes.detach())
                        d_real = disc(images)
                        g_loss = gen_loss(d_fake)
                        d_loss = disc_loss(d_real, d_fake)

                


                    sw.add_scalar("validation/fake_image_prediction",torch.mean(d_fake).item(),steps)
                    sw.add_scalar("validation/real_image_prediction",torch.mean(d_real).item(),steps)
                    sw.add_scalar("validation/generator_loss",d_loss/16,steps)

                    sw.add_scalar("validation/discriminator_loss",d_loss/16,steps)
                    sw.add_images("validation/greyscale images", bws[:, None, :, : ],steps)
                    sw.add_images("validation/real_images", images,steps)
                    sw.add_images("validation/generated_images", generated,steps)



                
                torch.save({
                    'steps': steps,
                    'generator': gen.state_dict(),
                    'generator_optimizer': g_optim.state_dict(),
                    'discriminator': disc.state_dict(),
                    'discriminator_optimizer': d_optim.state_dict(),
                    'generator_loss': g_loss,
                    'discriminator_loss': d_loss
                    }, path)
                
            steps +=1
            gen.train()
            disc.train()
        sw.add_scalar("training/generator_loss",epoch_g_loss/len(train_ds),epoch)
        sw.add_scalar("training/discriminator_loss",epoch_d_loss/len(train_ds),epoch)
        
  

if __name__== '__main__':
    main()

