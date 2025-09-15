# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
from config import Config
from dataset import MalayalamTTSDataset, collate_fn

class IndicF5Trainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize wandb
        wandb.init(
            project="indicf5-malayalam-tts",
            config=vars(config)
        )
        
        # Load model
        print("Loading IndicF5 model...")
        self.model = AutoModel.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        self.model.to(config.device)
        
        # Load datasets
        print("Loading datasets...")
        self.train_dataset = MalayalamTTSDataset(
            config.data_dir, 
            split="train", 
            config=config
        )
        self.val_dataset = MalayalamTTSDataset(
            config.data_dir, 
            split="test", 
            config=config
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            texts = batch['texts']
            audios = batch['audios'].to(self.config.device)
            
            # Forward pass through IndicF5
            # Note: You may need to adapt this based on IndicF5's training interface
            try:
                # This is a placeholder - you'll need to implement the actual training forward pass
                # The exact implementation depends on IndicF5's internal training methods
                loss = self.compute_loss(texts, audios)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # Logging
                if step % self.config.logging_steps == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": step
                    })
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def compute_loss(self, texts, audios):
        """
        Compute training loss for IndicF5
        Note: This needs to be implemented based on IndicF5's architecture
        """
        # Placeholder implementation
        # You'll need to adapt this based on how IndicF5 handles training
        
        # Option 1: If IndicF5 has a built-in training method
        # loss = self.model.training_step(texts, audios)
        
        # Option 2: If you need to implement custom loss
        # generated_audio = self.model.generate(texts)
        # loss = nn.MSELoss()(generated_audio, audios)
        
        # For now, return a dummy loss
        return torch.tensor(0.1, requires_grad=True)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                texts = batch['texts']
                audios = batch['audios'].to(self.config.device)
                
                try:
                    loss = self.compute_loss(texts, audios)
                    total_loss += loss.item()
                except:
                    continue
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_model(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        print("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss_epoch": train_loss,
                "val_loss_epoch": val_loss
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch, val_loss)

def main():
    config = Config()
    trainer = IndicF5Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
