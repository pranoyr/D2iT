
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import wandb
from tqdm import tqdm
# import constant_learnign rate swith warm up
import logging
import math
import cv2

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from einops import rearrange
import logging





from d2it.dataset import get_loaders 
from d2it.dgt import DiT_S_2, DiT_L_8




def resume_from_checkpoint(device, filename, model, optim, scheduler):
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        global_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        
        logging.info(f"Resumed from checkpoint: {filename} at step {global_step}")

        return global_step


def save_ckpt(accelerator, model, optim, scheduler, global_step, filename):
    checkpoint={
            'step': global_step,
            'model_state_dict': accelerator.get_state_dict(model),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
    accelerator.save(checkpoint, filename)
    logging.info("Saving checkpoint: %s ...", filename)


@torch.no_grad()
def sample_images(model, val_dl, accelerator, device, global_step):
    model.eval()
    
    # Get a batch of validation images
    val_batch = next(iter(val_dl))
    val_images = val_batch.to(device)
    
    # Take only first 8 samples for visualization
    n_samples = min(8, val_images.shape[0])
    val_images = val_images[:n_samples]
    
    # Get model output
    model_output = model(val_images)
    reconstructed = model_output['reconstructed']
    
    # Normalize images for visualization (assuming input is [-1,1])
    val_images_viz = (val_images + 1.0) / 2.0
    reconstructed_viz = (reconstructed + 1.0) / 2.0
    
    # Clamp to valid range [0,1]
    val_images_viz = torch.clamp(val_images_viz, 0, 1)
    reconstructed_viz = torch.clamp(reconstructed_viz, 0, 1)
    
    # Create comparison grid: original | reconstructed
    comparison = torch.cat([val_images_viz, reconstructed_viz], dim=0)
    grid = make_grid(comparison, nrow=n_samples, padding=2, normalize=False)
    
    # Log to wandb if main process
    if accelerator.is_main_process:
        accelerator.log({
            "reconstructions": wandb.Image(grid, caption=f"Step {global_step}: Top=Original, Bottom=Reconstructed")
        }, step=global_step)
    
    model.train()  # Set back to training mode
    


@torch.no_grad()
def validate(model, val_dl, loss_fn, weight_dict, accelerator, device, global_step):
    model.eval()

    total_loss, total_l1, total_l2, total_perc = 0, 0, 0, 0
    num_batches = 0

    for batch in tqdm(val_dl, desc="Validating", disable=not accelerator.is_main_process):
        images = batch.to(device)
        with accelerator.autocast():
            recon = model(images)
            loss, loss_dict = loss_fn(recon, images, weights=weight_dict)

        total_loss += loss_dict['total_loss']
        total_l1 += loss_dict['l1_loss']
        total_l2 += loss_dict['l2_loss']
        total_perc += loss_dict['perceptual_loss']
        num_batches += 1

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_l1 = total_l1 / num_batches
    avg_l2 = total_l2 / num_batches
    avg_perc = total_perc / num_batches

    if accelerator.is_main_process:
        accelerator.log({
            'val_total_loss': avg_loss,
            'val_l1_loss': avg_l1,
            'val_l2_loss': avg_l2,
            'val_perceptual_loss': avg_perc,
        }, step=global_step)

    model.train()



def train(args):

    global_step = 0

    # setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb")

    accelerator.init_trackers(
            project_name=args.project_name,
            # add kwargs for wandb
            init_kwargs={"wandb": {
                "config": vars(args)
            }}	
    )

    # set device
    device = accelerator.device
    # model
    model = DiT_S_2()
    # Train loders
    train_dl, val_dl = get_loaders(
        root_dir=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


    # training parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )
    steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    num_training_steps = args.num_epochs * steps_per_epoch
   
    # scheduler = get_cosine_schedule_with_warmup(
    #         optim,
    #         num_warmup_steps=args.warmup_steps,
    #         num_training_steps=num_training_steps
    #     )


    # scheduler = get_constant_schedule_with_warmup(
    #         optim,
    #         num_warmup_steps=args.warmup_steps
    #     )

    scheduler = None
      

    # prepare model, optimizer, and dataloader for distributed training
    model, optim, scheduler, train_dl, val_dl = accelerator.prepare(
        model, 
        optim, 
        scheduler, 
        train_dl, 
        val_dl
    )
    

    # load models
    if args.resume:
        global_step = resume_from_checkpoint(device, args.resume, model, optim, scheduler)

    effective_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    effective_training_steps = args.num_epochs * effective_steps_per_epoch

    logging.info(f"Effective batch size per device: {args.batch_size * args.gradient_accumulation_steps}")
    logging.info(f"Effective steps per epoch: {effective_steps_per_epoch}")
    logging.info(f"Effective Total training steps: {effective_training_steps}")

    start_epoch = global_step // effective_training_steps
    # start_epoch = 0

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        with tqdm(train_dl, dynamic_ncols=True, disable=not accelerator.is_main_process) as train_dl:
            for batch in train_dl:
                grain_map, labels = batch


                # print("Grain map shape:", grain_map.shape)
                # print("Labels shape:", labels.shape)

                # print(labels)
                # print(grain_map)
                # exit()

                        
                # =========================
                # GENERATOR TRAINING STEP
                # =========================
                with accelerator.accumulate(model):
                    
                    optim.zero_grad(set_to_none=True)
                    
                    with accelerator.autocast():
                        loss_dict = model(labels, target=grain_map)
                        loss = sum(loss_dict.get(f"{k}_weight", 1.0) * loss_dict[k] for k in loss_dict if 'weight' not in k)

                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients and args.max_grad_norm:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optim.step()	
                    if scheduler is not None:
                        scheduler.step()

            
                # =========================
                # LOGGING AND CHECKPOINTING
                # =========================
                if accelerator.sync_gradients:
                    # if not (global_step % args.ckpt_every) and accelerator.is_main_process:
                    #     save_ckpt(accelerator,
                    #               model,
                    #               optim,
                    #               scheduler,
                    #               global_step,
                    #               os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth'))
                    

                    # if not (global_step % args.eval_every):
                    #     validate(
                    #         model,
                    #         val_dl,
                    #         loss_fn,
                    #         accelerator,
                    #         device,
                    #         global_step
                    #     )
                    
                    # Prepare logging
                    log_dict = {
                        "lr": optim.param_groups[0]['lr'],
                    }

                    # Loop over loss_dict and add all losses
                    for key, value in loss_dict.items():
                        if 'weight' not in key:  # Skip weight entries
                            log_dict[key] = value.item() if torch.is_tensor(value) else value
                    
                    # add total also
                    log_dict["total_loss"] = loss.item()


                    accelerator.log(log_dict, step=global_step)
                    global_step += 1

    accelerator.end_training()        
    print("Train finished!")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()

    # project / dataset
    parser.add_argument('--project_name', type=str, default='Grain Transformer')
    parser.add_argument('--root', type=str, default='/media/pranoy/Datasets/ILSVRC/grain_maps',help="Path to dataset")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4 , help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="LR warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help="Mixed precision training mode")


    # logging / checkpointing
    parser.add_argument('--ckpt_every', type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument('--eval_every', type=int, default=2000, help="Evaluate every N steps")
    parser.add_argument('--ckpt_saved_dir', type=str, default='ckpt/d_grain_transformer', help="Directory to save outputs")

    args = parser.parse_args()


    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
 

    kwargs = vars(args)
    print("Training configuration:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    train(args)
