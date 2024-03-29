import numpy as np
import torch
import torch.nn as nn
from models import Generator, Discriminator,gen_loss, disc_loss
from tqdm import tqdm
import cv2
from utils import AHE, FFHQ, collate
import lazy_dataset
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sw = SummaryWriter()
#load the files
ex = Experiment('pix2pix_colourisation',save_git_info=False)


@ex.config
def defaults():
    batch_size = 16
    d_lr = 0.00005
    g_lr = 0.0001
    steps_per_eval = 1000 # after how many steps to evluate the model and add images to the tensorboard
    use_hq = True # True for FFHQ False for AHE
    use_fp16 = True # True for half precision, useful for smaller GPUs

@ex.capture
def load_img(example, use_hq):
    bw = cv2.imread(example['image_path'], 0).astype(np.float32) / 255.0
    img = cv2.imread(example['image_path']).astype(np.float32) / 255.0
    if use_hq:
        #due to limited hardware, had to downsize the images from 1024x1024 to 512x512
        bw = cv2.resize(bw,(512,512))
        img = cv2.resize(img,(512,512))
        img = np.flip(img, axis=-1) # BGR2RGB and vice versa, needed for FFHQ, faster and easier than cv2's cvtcolor
    example['image'] = np.array([bw,bw,bw])
    example['target'] = img

    return example



@ex.capture
def prepare_dataset(dataset,batch_size):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size, drop_last=True)
    dataset = dataset.map(collate)
    return dataset

@ex.automain
def main(batch_size,d_lr,g_lr, steps_per_eval,use_hq,use_fp16):
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5
    scaler = torch.cuda.amp.GradScaler()

    if use_hq:
        db = FFHQ()
        path = 'ckpt_FFHQ.pth'

    else:
        db = AHE()
        path = 'ckpt_AHE.pth'

    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    steps = 0
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.train()
    disc.train()
    g_optim = torch.optim.Adam(gen.parameters(),lr=g_lr)
    d_optim = torch.optim.Adam(disc.parameters(), lr=d_lr)
    aux_criterion = nn.L1Loss()
    CE  = nn.BCELoss()
    for epoch in range(10000):
        epoch_g_loss = 0
        epoch_d_loss = 0
        train_ds = prepare_dataset(t_ds)
        valid_ds = prepare_dataset(v_ds,batch_size=1)
        for index,batch in enumerate(tqdm(train_ds)):
            images = batch['target']
            bws = batch['image']

            # cv2 loads images as (h,w,3), models take(3,h,w)
            images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2) 
            bws = torch.tensor(np.array(bws)).to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                fakes = gen(bws)
    


            #*********************  
            
            # discriminator step

            #*********************
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                    d_real = disc(images)
                    with torch.no_grad():
                        d_fake = disc(fakes)
                    loss_d = disc_loss(d_real, d_fake) * 0.5

                    scaler.scale(loss_d).backward()
                    scaler.step(d_optim)
                    scaler.update()
              
                    epoch_d_loss += loss_d.item()
                    d_optim.zero_grad()


            #******************* 
            # generator step
        
            #*******************
            with torch.cuda.amp.autocast(enabled=use_fp16):
                d_real = disc(images)
                fakes = fakes.requires_grad_(True)
                d_fake = disc(fakes)
                loss_g = gen_loss(d_fake) + aux_criterion(fakes, images) * 100
                scaler.scale(loss_g).backward()
                scaler.step(g_optim)
                scaler.update()
                epoch_g_loss += loss_g.item()
                g_optim.zero_grad()
                
           
            if steps % steps_per_eval == 0:
                gen.eval()
                disc.eval()
                for batch in tqdm(valid_ds[0:2]):
                    images = batch['target']
                    bws = batch['image']
                    images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2)
                    bws = torch.tensor(np.array(bws)).to(device)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=False):
                            generated = gen(bws)
                            d_fake = disc(generated)
                            d_real = disc(images)
                            g_loss = gen_loss(d_fake)
                            d_loss = disc_loss(d_real, d_fake)

            


                    sw.add_scalar("validation/fake_image_prediction",torch.mean(d_fake).item(),steps)
                    sw.add_scalar("validation/real_image_prediction",torch.mean(d_real).item(),steps)
                    sw.add_images("validation/greyscale images", bws,steps)
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
        
  