import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.model import AdaptorEncoder

from ldm.data.fundus import fundus, DRIVE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random



def seed_everything(seed=23):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="/home/lhq323/Projects/0_ldm_save/reconstraction_AE_likeDRIVE1000_kaggle_5e-4/configs/2023-12-01T14-18-49-project.yaml")
    parser.add_argument('--ckpt_dir', default="/home/lhq323/Projects/0_ldm_save/reconstraction_AE_likeDRIVE1000_kaggle_5e-4/checkpoints/epoch=000180.ckpt")
    parser.add_argument('--output_dir', default="/home/lhq323/Projects/0_ldm_save/test")
    parser.add_argument('--train_root_blur', default="/home/lhq323/Projects/0_Hanhn/dataset/from_GAMMA/train_contrast")
    parser.add_argument('--train_root_clear', default="/home/lhq323/Projects/0_Hanhn/dataset/from_GAMMA/train")
    parser.add_argument('--valid_root_blur', default="/home/lhq323/Projects/0_Hanhn/dataset/from_GAMMA/test_contrast")
    parser.add_argument('--valid_root_clear', default="/home/lhq323/Projects/0_Hanhn/dataset/from_GAMMA/test")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpu_id', default='0')
    args = parser.parse_args()
    return args

def get_dataset(opt, config):
    # train_dataset = fundus(opt.train_root)
    # valid_dataset = fundus(opt.valid_root, image_number=5)
    train_dataset = DRIVE(opt.train_root_blur, opt.train_root_clear)
    valid_dataset = DRIVE(opt.valid_root_blur, opt.valid_root_clear, image_number=5)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    return train_dataloader, valid_dataloader

def get_networks(opt, config):
    model = instantiate_from_config(config.model)
    ckpt = torch.load(opt.ckpt_dir, map_location='cpu')["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    original_encoder = model.encoder
    original_encoder_state_dict = original_encoder.state_dict()
    adaptor_encoder = AdaptorEncoder(**config.model.params.ddconfig)
    adaptor_encoder.load_state_dict(original_encoder_state_dict, strict=False)

    original_encoder.eval()
    for k, v in adaptor_encoder.named_parameters():
        if 'adaptor' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False
    
    # 保存初始化后的模型
    os.makedirs(opt.output_dir, exist_ok=True)
    initial_model_path = os.path.join(opt.output_dir, 'ours_initial.ckpt')
    torch.save({'state_dict': model.state_dict()}, initial_model_path)
    print(f"Initial model saved to {initial_model_path}")
    
    return original_encoder, adaptor_encoder, model

def get_input(batch, device):
    blur = batch["blur"]
    blur = blur.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    clear = batch["clear"]
    clear = clear.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    blur = blur.to(device)
    clear = clear.to(device)
    return blur, clear

if __name__ == '__main__':
    seed_everything(23)
    opt = get_parser()
    device = torch.device("cuda:"+opt.gpu_id)
    config = OmegaConf.load(opt.config_dir)
    train_dataloader, valid_dataloader = get_dataset(opt, config)
    original_encoder, adaptor_encoder, model = get_networks(opt, config)
    original_encoder.to(device)
    original_encoder.eval()
    adaptor_encoder.to(device)
    model.to(device)
    model.eval()

    criterion_L1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adaptor_encoder.parameters()), lr = config.model.base_learning_rate)

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'images'), exist_ok=True)

    for epoch in range(opt.epochs):
        adaptor_encoder.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        sum_loss = 0
        for idx, batch in pbar:
            blur, clear = get_input(batch, device)
            latent_clear_original = original_encoder(clear)
            latent_clear_original = latent_clear_original.detach()
            latent_blur_adaptor = adaptor_encoder(blur)
            latent_clear_adaptor = adaptor_encoder(clear)
            l1loss = criterion_L1(latent_clear_original, latent_blur_adaptor) + criterion_L1(latent_clear_original, latent_clear_adaptor)
            optimizer.zero_grad()
            l1loss.backward()
            optimizer.step()

            pbar.set_description(f'Epoch [{epoch}/{opt.epochs}]')
            sum_loss = l1loss.item() + sum_loss
            ave_loss = sum_loss / (idx + 1)
            pbar.set_postfix(loss=ave_loss)
        
        adaptor_encoder.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                blur, _ = get_input(batch, device)
                h = adaptor_encoder(blur)
                h = model.quant_conv(h)
                quant, emb_loss, info = model.quantize(h)
                enhanced = model.decode(quant)

                enhanced = torch.clamp((enhanced+1.0)/2.0, min=0.0, max=1.0)
                enhanced = enhanced.detach().cpu().numpy().transpose(0,2,3,1)[0]*255

                image_name = os.path.split(batch['blur_path'][0])[1][:-4]
                out_path = os.path.join(opt.output_dir, 'images', image_name + '_' + str(epoch) + '.png')
                Image.fromarray(enhanced.astype(np.uint8)).save(out_path)
        torch.cuda.empty_cache()

        if (epoch + 1)  % 10 == 0:
            model.encoder = adaptor_encoder.cpu()
            torch.save({'state_dict': model.state_dict()}, os.path.join(opt.output_dir, str(epoch)+'_ftAE.ckpt'))
            model.to(device)











# class fineTuneAEModel(pl.LightningModule):
#     def __init__(self, original_encoder, adaptor_encoder, config):
#         super(pl.LightningModule, self).__init__()
#         self.original_encoder = original_encoder
#         self.original_encoder.eval()
#         self.adaptor_encoder = adaptor_encoder
#         self.config = config
#         for k, v in adaptor_encoder.named_parameters():
#             if 'adaptor' in k:
#                 v.requires_grad = True
#             else:
#                 v.requires_grad = False
#         self.loss = torch.nn.L1Loss()
    
#     def forward(self, blur, clear):
#         latent_clear_original = self.original_encoder(clear)
#         latent_blur_adaptor = self.adaptor_encoder(blur)
#         return latent_clear_original, latent_blur_adaptor


#     def get_input(self, batch):
#         blur = batch["blur"]
#         blur = blur.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
#         clear = batch["clear"]
#         clear = clear.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
#         return blur, clear


#     def training_step(self, batch, batch_idx):
#         blur, clear = self.get_input(batch)
#         latent_clear_original, latent_blur_adaptor = self(blur, clear)
#         l1loss = self.loss(latent_clear_original.detach(), latent_blur_adaptor)
#         self.log("train_loss", l1loss)
#         return l1loss
    
#     def configure_optimizers(self):
#         # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.adaptor_encoder.parameters()), lr=self.config.model.base_learning_rate)
#         optimizer =  torch.optim.Adam(self.adaptor_encoder.parameters(), lr=self.config.model.base_learning_rate)
#         return optimizer
    

# checkpoint_callback = ModelCheckpoint(
#     dirpath = "./logs/finetuneAE/",
#     # save_last = True,
#     every_n_epochs = 10
# )

# trainer = pl.Trainer(max_epochs=30, gpus="0,")

# ftmodel = fineTuneAEModel(original_encoder, adaptor_encoder, config)
# trainer.fit(ftmodel, train_dataloader)

