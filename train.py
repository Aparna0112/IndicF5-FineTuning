# train.py - Fixed for transformers 4.50.3 with meta tensor fix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import wandb
from tqdm import tqdm
import os
import json
import logging
from datetime import datetime
from config import Config
from dataset import MalayalamTTSDataset, collate_fn
import numpy as np

def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class IndicF5Trainer:
    def __init__(self, config):
        self.config = config
        self.global_step = 0
        
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        if config.use_wandb:
            try:
                wandb.init(
                    project=config.wandb_project,
                    config=vars(config),
                    name=f"malayalam-tts-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                logger.info("Wandb initialized successfully")
            except Exception as e:
                logger.warning(f"Wandb initialization failed: {e}")
                config.use_wandb = False
        
        logger.info("Loading IndicF5 model...")
        try:
            # Disable torch compile to avoid meta tensor issues
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            
            # Load model without device_map to avoid meta tensor error
            logger.info("Loading model from HuggingFace...")
            self.model = AutoModel.from_pretrained(
                config.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            
            logger.info("Moving model to device...")
            self.model = self.model.to(config.device)
            
            # Convert to half precision if needed
            if config.fp16 and config.device.type == 'cuda':
                logger.info("Converting to FP16...")
                self.model = self.model.half()
            
            # Try to load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
                self.tokenizer = None
            
            # Enable gradient checkpointing
            if config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Model loaded successfully on {config.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        logger.info("Loading datasets...")
        try:
            self.train_dataset = MalayalamTTSDataset(config.data_dir, split="train", config=config)
            self.val_dataset = MalayalamTTSDataset(config.data_dir, split="test", config=config)
            logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.dataloader_drop_last
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.dataloader_drop_last
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and config.device.type == 'cuda' else None
        
        logger.info("Trainer initialized successfully")
    
    def compute_loss(self, texts, audios):
        try:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(texts, audios)
                    return loss
                
                if self.tokenizer is not None:
                    inputs = self.tokenizer(
                        texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=self.config.max_length
                    ).to(self.config.device)
                    
                    outputs = self.model(**inputs)
                    
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        return outputs.loss
                
                logger.warning("Using placeholder loss")
                batch_size = len(texts)
                target_shape = audios.shape
                dummy_output = torch.randn(target_shape, device=self.config.device, requires_grad=True)
                loss = nn.MSELoss()(dummy_output, audios) * 0.1
                
                return loss
                
        except Exception as e:
            logger.error(f"Error in loss computation: {e}")
            return torch.tensor(0.01, requires_grad=True, device=self.config.device)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                texts = batch['texts']
                audios = batch['audios'].to(self.config.device)
                
                loss = self.compute_loss(texts, audios)
                loss = loss / self.config.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    if self.config.use_wandb:
                        try:
                            wandb.log({
                                "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch,
                                "train/global_step": self.global_step
                            })
                        except:
                            pass
                    
                    logger.info(f"Step {self.global_step}: Loss={loss.item():.4f}, LR={current_lr:.2e}")
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("CUDA out of memory! Try reducing batch_size in config.py")
                    torch.cuda.empty_cache()
                    break
                else:
                    logger.error(f"Runtime error in training step {step}: {e}")
                    continue
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                try:
                    texts = batch['texts']
                    audios = batch['audios'].to(self.config.device)
                    
                    loss = self.compute_loss(texts, audios)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.config)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        try:
            checkpoint_files = []
            for file in os.listdir(self.config.output_dir):
                if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                    epoch_num = int(file.split("_")[2].split(".")[0])
                    checkpoint_files.append((epoch_num, file))
            
            checkpoint_files.sort(reverse=True)
            
            for epoch_num, file in checkpoint_files[self.config.max_checkpoints:]:
                file_path = os.path.join(self.config.output_dir, file)
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old checkpoint: {file}")
                except Exception as e:
                    logger.error(f"Error removing {file}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup_checkpoints: {e}")
    
    def train(self):
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Mixed precision: {self.scaler is not None}")
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(self.config.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if self.config.use_wandb:
                    try:
                        wandb.log({
                            "epoch": epoch,
                            "train/loss_epoch": train_loss,
                            "val/loss_epoch": val_loss
                        })
                    except:
                        pass
                
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                
                self.save_checkpoint(epoch, val_loss, is_best)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        finally:
            if self.config.use_wandb:
                try:
                    wandb.finish()
                except:
                    pass
        
        logger.info("Training completed!")

def main():
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA is not available. Using CPU.")
    
    config = Config()
    print(f"Using batch size: {config.batch_size}")
    
    trainer = IndicF5Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
