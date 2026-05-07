import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import logging
import platform
import psutil
import time
from pathlib import Path
from tqdm import tqdm

class DenoisingTrainer:
    def __init__(self, **cfg):
        self.args = DictConfig(cfg)

        # Core setup
        self.setup_logging()
        self.setup_device()
        self.setup_random_seeds()
        self.setup_mixed_precision()

        # Component configuration
        self.config_model()
        self.config_optimizer()
        self.config_loss()
        self.config_metrics()
        self.config_dataset()
        self.config_logger()

        # Final setup
        self.setup_checkpoint_manager()
        self.run_sanity_checks()


    def setup_logging(self):
        """Setup logging with error handling"""
        try:
            # Disable logging if specified
            if getattr(self.args, 'disable_logging', False):
                logging.disable(logging.CRITICAL)
                self.log = None
                return

            # Setup basic logging
            logging.basicConfig(
                level=getattr(self.args, 'log_level', logging.INFO),
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.log = logging.getLogger(__name__)

            print(f"Logger ready: {type(self.log).__name__}")
        except Exception as e:
            print(f"Logger failed: {e}")
            self.log = None

    def setup_device(self):
        """Setup CUDA/CPU device"""
        if torch.cuda.is_available():
            self.args.device = f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.args.device = 'cpu'
            print('Using CPU')

    def setup_random_seeds(self):
        """Set random seeds for reproducibility"""
        if hasattr(self.args, 'seed') and self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"All random seeds set to: {self.args.seed}")
        else:
            print('No seed provided, using random seed')

    def setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.args.use_mixed_precision = getattr(self.args, 'use_mixed_precision', False)

        if self.args.use_mixed_precision:
            if not torch.cuda.is_available():
                print("Warning: Mixed precision requested but CUDA not available")
                self.args.use_mixed_precision = False
                return

        if self.args.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def config_loss(self):
        """Configure loss function using pure Hydra"""
    

        # Pure Hydra instantiation
        self.criterion = hydra.utils.instantiate(self.args.loss)

        # Move to device if it's a module
        if isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion.to(self.args.device)

        print(f"Created loss function: {self.criterion.__class__.__name__}")

    def setup_checkpoint_manager(self):
        """Setup checkpoint management"""
        print("Configuring checkpoint management...")

        # Import and create checkpoint manager
        from utils.checkpoint_manager import CheckpointManager

        # Setup paths
        self.work_dir = Path.cwd()
        self.checkpoint_dir = self.work_dir / "checkpoints"
        self.model_dir = self.work_dir / "models"

        print(f"Working directory: {self.work_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Model directory: {self.model_dir}")

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.args)

        # Load checkpoint if specified
        if hasattr(self.args, 'resume_from') and self.args.resume_from:
            print(f"Loading checkpoint: {self.args.resume_from}")
            checkpoint = self.checkpoint_manager.load_checkpoint(self.args.resume_from)
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Store the starting epoch for resumption
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed from epoch {checkpoint['epoch']}, will continue from epoch {self.start_epoch}")
            else:
                print("Failed to load checkpoint")
                self.start_epoch = 0
        else:
            print("No checkpoint loading configured")
            self.start_epoch = 0

    def run_sanity_checks(self):
        """Run system and configuration sanity checks"""
        print("Running sanity checks...")

        # System info
        self.print_system_info()

        # Model summary
        self.get_model_summary()

        # Dataset statistics
        self.check_dataset_statistics()

        # Training configuration summary
        self.print_training_config()

        # Training time estimation
        self.estimate_training_time()

    def print_system_info(self):
        """Print system information"""
        print("=== System Information ===")
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print("=" * 50)

    def get_model_summary(self):
        """Print model summary"""
        print("=== Model Summary ===")
        print(f"Model: {self.model.__class__.__name__}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("=" * 50)

    def check_dataset_statistics(self):
        """Print dataset statistics"""
        print("=== Dataset Statistics ===")

        if hasattr(self, 'trainDataset'):
            print(f"Training samples: {len(self.trainDataset):,}")
            if hasattr(self, 'train_loader'):
                print(f"Training batches per epoch: {len(self.train_loader):,}")

        if hasattr(self, 'evalDataset'):
            print(f"Evaluation samples: {len(self.evalDataset):,}")
            if hasattr(self, 'eval_loader'):
                print(f"Evaluation batches: {len(self.eval_loader):,}")

        print("=" * 50)

    def print_training_config(self):
        """Print training configuration summary"""
        print("\n" + "=" * 60)
        print("TRAINING CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Epochs: {getattr(self.args, 'num_epochs', 'Not set')}")
        print(f"Device: {self.args.device}")
        print(f"Mixed Precision: {self.args.use_mixed_precision}")
        print(f"Gradient Clipping: {getattr(self.args, 'grad_clip', 'Not set')}")
        print(f"Early Stopping: {getattr(self.args, 'early_stopping', False)}")
        print(f"Checkpoint Freq: {getattr(self.args, 'checkpoint_freq', 'Not set')}")
        print(f"Use Validation: {getattr(self.args, 'use_validation', False)}")
        print(f"Use Metrics: {getattr(self.args, 'use_metrics', False)}")
        print("=" * 60)

    def estimate_training_time(self):
        """Estimate training time"""
        print("\n=== Training Time Estimation ===")

        if hasattr(self, 'train_loader'):
            batches_per_epoch = len(self.train_loader)
            total_epochs = getattr(self.args, 'num_epochs', 50)
            total_batches = batches_per_epoch * total_epochs

            # Rough estimate: 0.5 seconds per batch (very rough)
            estimated_seconds = total_batches * 0.5
            estimated_hours = estimated_seconds / 3600

            print(f"Batches per epoch: {batches_per_epoch:,}")
            print(f"Total batches for {total_epochs} epochs: {total_batches:,}")
            print(f"Estimated training time: {estimated_hours:.1f} hours")
            print("(This is a rough estimate based on average batch processing time)")

        print("=" * 50)

    def config_model(self):
        """Configure model using pure Hydra"""

        # Pure Hydra instantiation
        self.model = hydra.utils.instantiate(self.args.model)

        # Move to device
        self.model = self.model.to(self.args.device)

        # Handle DataParallel
        if getattr(self.args, 'use_data_parallel', False) and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")


    def config_optimizer(self):
        """Configure optimizer and scheduler"""

        # Pure Hydra instantiation for optimizer
        self.optimizer = hydra.utils.instantiate(
            self.args.optimizer,
            params=self.model.parameters()
        )

        # Configure scheduler if present
        if hasattr(self.args, 'scheduler') and self.args.scheduler is not None:
            scheduler_config = self.args.scheduler
            if hasattr(scheduler_config, '_target_') and scheduler_config._target_ is not None:
                self.scheduler = hydra.utils.instantiate(
                    scheduler_config,
                    optimizer=self.optimizer
                )
            else:
                self.scheduler = None
                print("No scheduler configured")
        else:
            self.scheduler = None
            print("Scheduler disabled")

        # Print summary
        self.print_optimizer_summary()



    def config_metrics(self):
        """Configure metrics - pure Hydra"""
        if not self.args.use_metrics:
            self.metrics = None
            self.val_metrics = None
            print("Metrics disabled")
            return

        # Pure Hydra instantiation
        self.metrics = hydra.utils.instantiate(self.args.metrics)

        # Move to device
        if hasattr(self.metrics, 'to'):
            self.metrics = self.metrics.to(self.args.device)

        # Always create separate validation metrics if validation is enabled
        # Sharing metrics between train and val contaminates the statistics
        if getattr(self.args, 'use_validation', False):
            self.val_metrics = hydra.utils.instantiate(self.args.metrics)
            if hasattr(self.val_metrics, 'to'):
                self.val_metrics = self.val_metrics.to(self.args.device)
            print("Created separate validation metrics")
        else:
            self.val_metrics = None
            print("Validation disabled, no validation metrics")


    def config_logger(self):
        """Configure logger using Hydra"""
        if hasattr(self.args, 'logger') and self.args.logger is not None:
            try:
                # Use pure Hydra instantiation with minimal overrides
                self.wandb_logger = hydra.utils.instantiate(
                    self.args.logger,
                    project=getattr(self.args, 'project_name', 'MultiViewLossProject'),
                    name=getattr(self.args, 'experiment_name', 'experiment')
                )
            except Exception as e:
                print(f"Warning: Logger setup failed: {e}")
                self.wandb_logger = None
        else:
            self.wandb_logger = None

    def config_dataset(self):
        """Configure datasets using Hydra instantiation"""

        # Setup collate function
        self.setup_collate_function()

        # Create datasets using pure Hydra instantiation
        if not self.args.test:
            self.trainDataset = hydra.utils.instantiate(self.args.dataset_train)

        if self.args.use_validation or self.args.test:
            self.evalDataset = hydra.utils.instantiate(self.args.dataset_test)

        # Create dataloaders using pure Hydra instantiation
        if hasattr(self, 'trainDataset'):
            self.train_loader = hydra.utils.instantiate(
                self.args.dataloader_train,
                dataset=self.trainDataset,
                collate_fn=self.collate_fn
            )

        if hasattr(self, 'evalDataset'):
            self.eval_loader = hydra.utils.instantiate(
                self.args.dataloader_eval,
                dataset=self.evalDataset,
                collate_fn=self.collate_fn
            )


        # Explicit target configuration - no auto-detection
        print(f"Target detection: hasTarget={self.args.hasTarget}")

    def setup_collate_function(self):
        """Setup collate function"""
        # Check dataset configuration for collate_batch parameter
        collate_fn_name = 'default'

        # Check train dataset config first
        if hasattr(self.args, 'dataset_train') and hasattr(self.args.dataset_train, 'args') and hasattr(self.args.dataset_train.args, 'collate_batch'):
            collate_fn_name = self.args.dataset_train.args.collate_batch
        # Fallback to general collate_fn parameter
        elif hasattr(self.args, 'collate_fn'):
            collate_fn_name = self.args.collate_fn

        if collate_fn_name == 'pad_list_data_collate':
            from monai.data import pad_list_data_collate
            self.collate_fn = pad_list_data_collate
        else:
            # Use default PyTorch collate function
            from torch.utils.data import default_collate
            self.collate_fn = default_collate
            print(f"Warning: Using default PyTorch collate function: {collate_fn_name}")

    def print_optimizer_summary(self):
        """Print optimizer summary"""
        print("\n=== Optimizer Summary ===")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")

        # Get learning rate
        if hasattr(self.optimizer, 'param_groups'):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {lr}")

            # Get weight decay if present
            if 'weight_decay' in self.optimizer.param_groups[0]:
                wd = self.optimizer.param_groups[0]['weight_decay']
                print(f"Weight decay: {wd}")

        # Scheduler info
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            print(f"Scheduler: {self.scheduler.__class__.__name__}")
        else:
            print("Scheduler: None")

        print("=" * 24)

    def setup_dataloaders(self):
        """Ultra-clean dataloader setup"""
        from torch.utils.data import DataLoader
        from monai.data import DataLoader as MonaiDataLoader

        DL_class = MonaiDataLoader if 'monai' in str(getattr(self.args, 'dataset', '')) else DataLoader

        # Create train dataloader
        if hasattr(self, 'trainDataset'):
            self.train_loader = self.create_with_matching_args(
                DL_class,
                required_args={'dataset': self.trainDataset, 'collate_fn': self.collate_fn},
                exclude=['self', 'dataset'],
                shuffle=getattr(self.args, 'shuffle', True)  # Override for training
            )

        # Create eval dataloader
        if hasattr(self, 'evalDataset') and self.evalDataset is not None:
            self.eval_loader = self.create_with_matching_args(
                DL_class,
                required_args={'dataset': self.evalDataset, 'collate_fn': self.collate_fn},
                exclude=['self', 'dataset'],
                shuffle=False)# No shuffling for evaluation


    def create_with_matching_args(self, cls, required_args=None, exclude=None, **overrides):
        """Helper to create objects with only matching arguments"""
        import inspect

        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys())

        # Remove excluded parameters
        if exclude:
            valid_params -= set(exclude)

        # Start with required args
        kwargs = required_args or {}

        # Add any args that match the constructor
        for key, value in self.args.items():
            if key in valid_params and key not in kwargs:
                kwargs[key] = value

        # Apply overrides
        kwargs.update(overrides)

        # Only keep valid parameters
        final_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        return cls(**final_kwargs)

    def reset_metrics(self, mode='train'):
        """Reset metrics for new epoch"""
        metrics = self.metrics if mode == 'train' else self.val_metrics
        if metrics is not None and hasattr(metrics, 'reset'):
            metrics.reset()

    def compute_metrics(self, outputs, targets, mode='train'):
        """Compute metrics"""
        metrics = self.metrics if mode == 'train' else self.val_metrics
        if metrics is not None:
            return metrics(outputs, targets)
        return {}

    def get_metrics(self, mode='train'):
        """Get computed metrics"""
        metrics = self.metrics if mode == 'train' else self.val_metrics
        if metrics is not None and hasattr(metrics, 'compute'):
            return metrics.compute()
        return {}

    def save_checkpoint(self, epoch, loss, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
        }

        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_manager.get_best_model_path()
            torch.save(checkpoint, best_path)
            print(f"💾 New best checkpoint saved: {best_path}")

        # Save epoch checkpoint
        if epoch % self.args.checkpoint_freq == 0:
            epoch_path = self.checkpoint_manager.get_checkpoint_path(epoch)
            torch.save(checkpoint, epoch_path)
            print(f"💾 Checkpoint saved: {epoch_path}")

        # Save latest checkpoint
        if getattr(self.args, 'save_last_checkpoint', True):
            latest_path = self.checkpoint_manager.get_latest_path()
            torch.save(checkpoint, latest_path)
            print(f"💾 Latest checkpoint updated: {latest_path}")

        # Cleanup old checkpoints if enabled
        if getattr(self.args, 'cleanup_checkpoints', False):
            max_keep = getattr(self.args, 'max_checkpoints_keep', 5)
            self.cleanup_old_checkpoints(max_keep)

    def cleanup_old_checkpoints(self, max_keep):
        """Remove old checkpoints, keeping only the last N"""
        if max_keep <= 0:
            return

        # Find all epoch checkpoints
        experiment_id = self.checkpoint_manager.experiment_id
        pattern = f"{experiment_id}_epoch_*.pth"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if len(checkpoints) > max_keep:
            # Sort by modification time (oldest first)
            checkpoints.sort(key=lambda x: x.stat().st_mtime)

            # Remove oldest checkpoints
            to_remove = checkpoints[:-max_keep]
            for checkpoint in to_remove:
                try:
                    checkpoint.unlink()
                    print(f"🗑️ Removed old checkpoint: {checkpoint.name}")
                except Exception as e:
                    print(f"Warning: Failed to remove {checkpoint.name}: {e}")

    def get_metric_value(self, metrics_dict, metric_name):
        """Extract metric value from metrics dictionary"""
        if metric_name == "val_loss" or metric_name == "train_loss":
            return metrics_dict.get(metric_name, float('inf'))

        # For other metrics, look in computed metrics
        val_metrics = self.get_metrics('val') if self.val_metrics else {}
        if metric_name in val_metrics:
            metric_value = val_metrics[metric_name]
            # Handle torch tensors
            if hasattr(metric_value, 'item'):
                return metric_value.item()
            return float(metric_value)

        return float('inf') if self.args.best_metric_mode == 'min' else float('-inf')

    def is_best_checkpoint(self, current_metrics):
        """Determine if current metrics represent the best model so far"""
        best_metric = getattr(self.args, 'best_metric', 'val_loss')
        best_mode = getattr(self.args, 'best_metric_mode', 'min')

        current_value = self.get_metric_value(current_metrics, best_metric)

        # Initialize best value if not exists
        if not hasattr(self, 'best_metric_value'):
            self.best_metric_value = float('inf') if best_mode == 'min' else float('-inf')

        # Check if current is better
        if best_mode == 'min':
            is_better = current_value < self.best_metric_value
        else:  # max
            is_better = current_value > self.best_metric_value

        if is_better:
            self.best_metric_value = current_value
            print(f"New best {best_metric}: {current_value:.4f}")

        return is_better

    def should_stop_early(self, current_metrics, epoch):
        """Check if training should stop early"""
        if not getattr(self.args, 'early_stopping', False):
            return False

        patience = getattr(self.args, 'early_stopping_patience', 10)
        best_metric = getattr(self.args, 'best_metric', 'val_loss')

        current_value = self.get_metric_value(current_metrics, best_metric)

        if not hasattr(self, 'early_stop_best_value'):
            self.early_stop_best_value = current_value
            self.patience_counter = 0
            return False

        best_mode = getattr(self.args, 'best_metric_mode', 'min')

        if best_mode == 'min':
            improved = current_value < self.early_stop_best_value
        else:
            improved = current_value > self.early_stop_best_value

        if improved:
            self.early_stop_best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= patience

    def train(self):
        """Main training loop"""
        # Check if we're in test-only mode
        if getattr(self.args, 'test', False) and not getattr(self.args, 'use_training', True):
            print("🧪 Running in test-only mode...")
            self.config_loss()
            self.setup_checkpoint_manager()
            # Run evaluation directly
            val_loss = self.validate_epoch(0)
            print(f"Test completed. Loss: {val_loss:.6f}")
            # Save test outputs after inference is complete
            if getattr(self.args, 'save_test_outputs', False):
                print("💾 Saving test outputs...")
                self.save_test_outputs()

            return


        # Re-configure for safety
        self.config_loss()
        self.setup_checkpoint_manager()

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Get starting epoch (0 for new training, or resume epoch)
        start_epoch = getattr(self, 'start_epoch', 0)

        for epoch in range(start_epoch, self.args.num_epochs):
            train_loss = self.train_epoch(epoch)

            val_loss = 0.0
            if self.args.use_validation:
                val_loss = self.validate_epoch(epoch)

            # Collect all metrics
            current_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss
            }

            # Add computed metrics if available
            if self.val_metrics is not None:
                val_metrics = self.get_metrics('val')
                current_metrics.update(val_metrics)

            # Update scheduler
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

            # Check if this is the best model using configurable metric
            is_best = self.is_best_checkpoint(current_metrics)

            if epoch % self.args.checkpoint_freq == 0:
                self.save_checkpoint(epoch, train_loss, current_metrics, is_best)

            # Print epoch info with metrics
            metric_str = f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            if self.val_metrics is not None:
                val_metrics = self.get_metrics('val')
                for name, value in val_metrics.items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    metric_str += f", {name}={value:.4f}"

            print(f"Epoch {epoch}: {metric_str}")

            # Log to WandB if available
            if hasattr(self, 'wandb_logger') and self.wandb_logger is not None:
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                log_dict = {
                    'epoch': epoch,
                    'learning_rate': current_lr,
                    'train/loss': train_loss,
                    'val/loss': val_loss
                }

                # Add training metrics
                if self.metrics is not None:
                    train_metrics = self.get_metrics('train')
                    for name, value in train_metrics.items():
                        if hasattr(value, 'item'):
                            value = value.item()
                        log_dict[f'train/{name}'] = value

                # Add validation metrics
                if self.val_metrics is not None:
                    val_metrics = self.get_metrics('val')
                    for name, value in val_metrics.items():
                        if hasattr(value, 'item'):
                            value = value.item()
                        log_dict[f'val/{name}'] = value

                # Log images if enabled and at the right frequency
                if (getattr(self.args, 'log_images_wandb', False) and
                    epoch % getattr(self.args, 'log_images_freq', 5) == 0):
                    image_logs = self.log_validation_images()
                    if image_logs:
                        log_dict.update(image_logs)

                try:
                    self.wandb_logger.log(log_dict)
                except Exception as e:
                    print(f"WandB logging failed: {e}")

            # Early stopping
            if self.should_stop_early(current_metrics, epoch):
                print(f"Early stopping at epoch {epoch}")
                break


    def train_epoch(self, epoch):
        """Single training epoch"""
        self.model.train()
        if self.metrics is not None:
            self.reset_metrics('train')

        total_loss = 0.0
        num_batches = 0


        train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch")

        for batch_idx, batch in enumerate(train_loader_tqdm):
            # Handle batch data
            if isinstance(batch, dict):
                inputs = batch['input'].to(self.args.device)
                targets = batch.get('target', batch.get('label')).to(self.args.device)
            else:
                inputs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device)

            # Training step
            self.optimizer.zero_grad()

            if self.args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if getattr(self.args, 'grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update metrics
            if self.metrics is not None:
                with torch.no_grad():
                    self.compute_metrics(outputs, targets, 'train')

        return total_loss / num_batches

    def validate_epoch(self, epoch):
        """Single validation epoch"""
        print(f"Starting validation for epoch {epoch}...")
        start_time = time.time()

        self.model.eval()
        if self.val_metrics is not None:
            self.reset_metrics('val')

        total_loss = 0.0
        num_batches = 0

        # Check if sliding window inference is enabled
        use_sliding_window = False
        if hasattr(self.args, 'dataset_test'):
            if hasattr(self.args.dataset_test, 'args'):
                use_sliding_window = getattr(self.args.dataset_test.args, 'use_sliding_window', False)
            else:
                use_sliding_window = getattr(self.args.dataset_test, 'use_sliding_window', False)

        with torch.no_grad():
            val_loader_tqdm = tqdm(self.eval_loader, desc=f"Validation {epoch}", unit="batch", leave=False)
            for batch in val_loader_tqdm:
                # Handle batch data
                if isinstance(batch, dict):
                    inputs = batch['input'].to(self.args.device)
                    targets = batch.get('target', batch.get('label')).to(self.args.device)
                else:
                    inputs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device)

                # Use sliding window inference if enabled
                if use_sliding_window:
                    outputs = self._sliding_window_inference(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                if self.val_metrics is not None:
                    self.compute_metrics(outputs, targets, 'val')

        validation_time = time.time() - start_time
        return total_loss / num_batches

    def _sliding_window_inference(self, inputs):
        """
        Perform sliding window inference on large volumes.

        Args:
            inputs: Input tensor (B, C, D, H, W) or (B, C, H, W)

        Returns:
            Output tensor of same shape as inputs
        """
        from monai.inferers import sliding_window_inference

        # Get sliding window parameters from config
        roi_size = getattr(self.args.dataset_test.args, 'roi_size', [64, 64, 64])
        sw_batch_size = getattr(self.args.dataset_test.args, 'sw_batch_size', 4)
        overlap = getattr(self.args.dataset_test.args, 'overlap', 0.5)

        # Convert roi_size to tuple if it's a list
        if isinstance(roi_size, list):
            roi_size = tuple(roi_size)

        # Perform sliding window inference
        outputs = sliding_window_inference(
            inputs=inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=overlap,
            mode='gaussian',  # Use gaussian weighting for blending
            device=self.args.device
        )

        return outputs

    def log_validation_images(self):
        """Log sample validation images to WandB"""
        try:
            import wandb
            import numpy as np
            import random

            if not hasattr(self, 'eval_loader') or self.eval_loader is None:
                return {}

            self.model.eval()
            image_logs = {}
            num_samples = min(getattr(self.args, 'num_vis_samples', 4), len(self.eval_loader))

            # Get batch indices from the middle portion of the dataset
            total_batches = len(self.eval_loader)
            middle_start = total_batches // 4  # Start from 25% through the dataset
            middle_end = 3 * total_batches // 4  # End at 75% through the dataset
            middle_range = list(range(middle_start, middle_end))

            batch_indices = random.sample(middle_range, min(num_samples, len(middle_range)))

            with torch.no_grad():
                for i, batch_idx in enumerate(batch_indices):
                    if i >= num_samples:
                        break

                    # Get the specific batch
                    batch = next(iter([batch for j, batch in enumerate(self.eval_loader) if j == batch_idx]))

                    # Handle batch data
                    if isinstance(batch, dict):
                        inputs = batch['input'].to(self.args.device)
                        targets = batch.get('target', batch.get('label')).to(self.args.device)
                    else:
                        inputs, targets = batch[0].to(self.args.device), batch[1].to(self.args.device)

                    # Run inference (use sliding window if enabled)
                    use_sliding_window = getattr(self.args.dataset_test.args, 'use_sliding_window', False)
                    if use_sliding_window:
                        outputs = self._sliding_window_inference(inputs)
                    else:
                        outputs = self.model(inputs)

                    # Take first sample from batch and remove batch dim
                    input_vol = inputs[0, 0].cpu().numpy()  # Shape: could be 2D [H, W] or 3D [D, H, W]
                    target_vol = targets[0, 0].cpu().numpy()
                    output_vol = outputs[0, 0].cpu().numpy()

                    # Check if data is 3D or 2D
                    is_3d = len(input_vol.shape) == 3

                    # Normalize images using percentile-based normalization for better contrast
                    def normalize_image(img):
                        # Use 1st and 99th percentile for more robust normalization
                        p1, p99 = np.percentile(img, [1, 99])
                        img_norm = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
                        return img_norm

                    # Ensure correct orientation for medical images
                    def fix_orientation(img):
                        # For sagittal views, we might need to rotate/flip
                        # Try different orientations to match typical medical image display
                        return np.rot90(img, k=-1)  # Rotate 90 degrees clockwise

                    if is_3d:
                        # Handle 3D volumes - extract middle slices from each view
                        D, H, W = input_vol.shape

                        # Axial view (slice along depth/D axis)
                        axial_slice_idx = D // 2
                        input_axial = input_vol[axial_slice_idx, :, :]
                        target_axial = target_vol[axial_slice_idx, :, :]
                        output_axial = output_vol[axial_slice_idx, :, :]

                        # Sagittal view (slice along width/W axis)
                        sagittal_slice_idx = W // 2
                        input_sagittal = input_vol[:, :, sagittal_slice_idx]
                        target_sagittal = target_vol[:, :, sagittal_slice_idx]
                        output_sagittal = output_vol[:, :, sagittal_slice_idx]

                        # Coronal view (slice along height/H axis)
                        coronal_slice_idx = H // 2
                        input_coronal = input_vol[:, coronal_slice_idx, :]
                        target_coronal = target_vol[:, coronal_slice_idx, :]
                        output_coronal = output_vol[:, coronal_slice_idx, :]

                        # Normalize and fix orientation for each view
                        for view_name, (inp, tgt, out) in [
                            ('axial', (input_axial, target_axial, output_axial)),
                            ('sagittal', (input_sagittal, target_sagittal, output_sagittal)),
                            ('coronal', (input_coronal, target_coronal, output_coronal))
                        ]:
                            inp_norm = normalize_image(inp)
                            tgt_norm = normalize_image(tgt)
                            out_norm = normalize_image(out)

                            inp_oriented = fix_orientation(inp_norm)
                            tgt_oriented = fix_orientation(tgt_norm)
                            out_oriented = fix_orientation(out_norm)

                            # Log images for this view
                            image_logs[f'val_images/{view_name}/sample_{i}_input'] = wandb.Image(inp_oriented, caption=f"Input {i} ({view_name})")
                            image_logs[f'val_images/{view_name}/sample_{i}_target'] = wandb.Image(tgt_oriented, caption=f"Target {i} ({view_name})")
                            image_logs[f'val_images/{view_name}/sample_{i}_output'] = wandb.Image(out_oriented, caption=f"Output {i} ({view_name})")

                    else:
                        # Handle 2D images (original behavior)
                        input_img = normalize_image(input_vol)
                        target_img = normalize_image(target_vol)
                        output_img = normalize_image(output_vol)

                        input_img = fix_orientation(input_img)
                        target_img = fix_orientation(target_img)
                        output_img = fix_orientation(output_img)

                        # Log individual images
                        image_logs[f'val_images/sample_{i}_input'] = wandb.Image(input_img, caption=f"Input {i}")
                        image_logs[f'val_images/sample_{i}_target'] = wandb.Image(target_img, caption=f"Target {i}")
                        image_logs[f'val_images/sample_{i}_output'] = wandb.Image(output_img, caption=f"Output {i}")

            return image_logs

        except Exception as e:
            print(f"Image logging failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def save_test_outputs(self):
        """Save test outputs after inference is complete"""
        import os
        import re
        import numpy as np
        import pydicom
        import nibabel as nib
        from pathlib import Path
        from collections import defaultdict


        # Get experiment name for output directory
        experiment_name = getattr(self.args, 'experiment_name', 'test_results')
        output_path = Path('test_results') / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect outputs and metadata by case
        case_outputs = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.eval_loader, desc="Saving outputs", unit="batch")):
                # Handle batch data
                if isinstance(batch, dict):
                    inputs = batch['input'].to(self.args.device)
                    # Get the data entry for metadata
                    data_entry = self.evalDataset.data[batch_idx]
                else:
                    inputs = batch[0].to(self.args.device)
                    data_entry = self.evalDataset.data[batch_idx]

                # Run inference
                outputs = self.model(inputs)

                # Extract case information from target path
                target_path = data_entry.get('target', '')

                # Try multiple patterns to extract case ID
                # Pattern 1: rectangular/8x/415/original_slices/...
                case_match = re.search(r'rectangular[/\\]\w+[/\\](\d+)[/\\]', target_path)
                if not case_match:
                    # Pattern 2: registration_C_anonymized/123/...
                    case_match = re.search(r'registration_C_anonymized[/\\](\d+)[/\\]', target_path)
                if not case_match:
                    # Pattern 3: registration_A_T1_slices/4x/1/...
                    case_match = re.search(r'registration_A_T1_slices[/\\]\w+[/\\](\d+)[/\\]', target_path)
                if not case_match:
                    # Pattern 4: registration_A_T1_resized/1/T1_HR.nii.gz (Clinical 3D volumes)
                    case_match = re.search(r'registration_A_T1_resized[/\\](\d+)[/\\]', target_path)

                case_id = case_match.group(1) if case_match else 'unknown'

                # Extract slice filename
                slice_filename = Path(target_path).name

                # Store output with metadata
                case_outputs[case_id].append({
                    'output': outputs.cpu().numpy(),
                    'filename': slice_filename,
                    'target_path': target_path,
                    'batch_idx': batch_idx
                })

        # Save outputs organized by case
        for case_id, case_data in case_outputs.items():

            print(f"Saving {len(case_data)} slices for case {case_id}...")

            # Check if we have NIfTI or DICOM by looking at first file
            first_target_file = Path(case_data[0]['target_path'])
            is_nifti = first_target_file.suffix in ['.nii', '.gz'] or str(first_target_file).endswith('.nii.gz')

            try:
                if is_nifti:
                    # Check if this is a 3D volume or 2D slices
                    # 3D volumes: "original.nii.gz" (IXI) or "T1_HR.nii.gz"/"BICUBIC_T1_LR.nii.gz" (Clinical)
                    first_filename = case_data[0]['filename']
                    is_3d_volume = (first_filename in ['original.nii.gz', 'T1_HR.nii.gz', 'BICUBIC_T1_LR.nii.gz'] and len(case_data) == 1)

                    if is_3d_volume:
                        # Handle 3D volume directly - already reconstructed
                        output_array = case_data[0]['output'][0, 0]  # Remove batch and channel dims
                        target_path = case_data[0]['target_path']

                        # Load original NIfTI to get affine and header
                        original_nii = nib.load(target_path)
                        affine = original_nii.affine
                        header = original_nii.header.copy()

                        # Update header dimensions
                        header.set_data_shape(output_array.shape)

                        # Create and save 3D NIfTI volume
                        output_nii = nib.Nifti1Image(output_array, affine, header)
                        output_file = output_path / f"{case_id}.nii.gz"
                        nib.save(output_nii, str(output_file))

                        print(f"  Saved 3D volume for case {case_id}: {output_array.shape} to {output_file}")

                    else:
                        # For 2D slices: Extract slice indices and sort
                        slice_data = []
                        orientation = None

                        for item in case_data:
                            output_array = item['output'][0, 0]  # Remove batch and channel dims
                            filename = item['filename']
                            target_path = item['target_path']

                            # Detect orientation from path (sagittal, axial, or coronal)
                            if 'sagittal' in target_path.lower():
                                orientation = 'sagittal'
                            elif 'axial' in target_path.lower():
                                orientation = 'axial'
                            elif 'coronal' in target_path.lower():
                                orientation = 'coronal'

                            # Extract slice index from filename (e.g., slice_064.nii.gz -> 64)
                            slice_match = re.search(r'slice_(\d+)', filename)
                            if slice_match:
                                slice_idx = int(slice_match.group(1))
                                slice_data.append({
                                    'index': slice_idx,
                                    'array': output_array,
                                    'path': item['target_path'],
                                    'filename': filename
                                })
                            else:
                                print(f"Warning: Could not extract slice index from {filename}")

                        # Sort slices by index
                        slice_data.sort(key=lambda x: x['index'])

                        # Stack all slices into 3D volume along the correct axis
                        volume_slices = [s['array'] for s in slice_data]
                        first_slice_shape = volume_slices[0].shape

                        # Detect Clinical/Sheba data by slice shape (128, 256) vs IXI (256, 128)
                        # Clinical slices need to be transposed to match IXI orientation before stacking
                        if first_slice_shape == (128, 256):
                            print(f"  Detected Clinical/Sheba slices {first_slice_shape}, transposing to match IXI orientation")
                            volume_slices = [np.transpose(s, (1, 0)) for s in volume_slices]
                            print(f"  Transposed slices to shape: {volume_slices[0].shape}")

                        # Use identical stacking logic for all datasets (IXI and Clinical)
                        # This ensures consistency since Clinical models are fine-tuned from IXI
                        # All datasets should produce (256, 256, 128) volumes
                        if orientation == 'sagittal':
                            volume_3d = np.stack(volume_slices, axis=2)
                        elif orientation == 'coronal':
                            volume_3d = np.stack(volume_slices, axis=0)
                        elif orientation == 'axial':
                            volume_3d = np.stack(volume_slices, axis=1)
                        else:
                            print(f"Warning: Unknown orientation '{orientation}', using axis=-1")
                            volume_3d = np.stack(volume_slices, axis=-1)

                        print(f"  Saved 3D {orientation} volume for case {case_id}: {volume_3d.shape}")

                        # Load first slice to get affine and header template
                        first_nii = nib.load(slice_data[0]['path'])
                        affine = first_nii.affine
                        header = first_nii.header.copy()

                        # Update header dimensions for 3D volume
                        header.set_data_shape(volume_3d.shape)

                        # Create and save 3D NIfTI volume with case_id as filename
                        output_nii = nib.Nifti1Image(volume_3d, affine, header)
                        output_file = output_path / f"{case_id}.nii.gz"
                        nib.save(output_nii, str(output_file))

                        print(f"  Saved 3D {orientation} volume for case {case_id}: {volume_3d.shape} to {output_file}")

                else:
                    # For DICOM: Save individual slices in case subdirectory
                    case_dir = output_path / case_id
                    case_dir.mkdir(exist_ok=True)

                    for item in case_data:
                        output_array = item['output'][0, 0]  # Remove batch and channel dims
                        filename = item['filename']

                        # Read original DICOM for metadata
                        original_dcm = pydicom.dcmread(item['target_path'])

                        # Get original pixel data to match format
                        original_pixel_array = original_dcm.pixel_array

                        # Convert output to same dtype as original
                        output_formatted = output_array.astype(original_pixel_array.dtype)

                        # Create a copy of the original DICOM to preserve metadata
                        output_dcm = pydicom.dcmread(item['target_path'])

                        # Update pixel data properly
                        output_dcm.PixelData = output_formatted.tobytes()
                        output_dcm.Rows, output_dcm.Columns = output_formatted.shape

                        # Ensure proper DICOM metadata consistency
                        if hasattr(output_dcm, 'PhotometricInterpretation'):
                            if len(output_formatted.shape) == 2:
                                output_dcm.PhotometricInterpretation = 'MONOCHROME2'

                        # Update image description
                        output_dcm.ImageComments = "Processed by neural network"

                        # Save to case subdirectory
                        output_file = case_dir / f"output_{filename}"
                        output_dcm.save_as(str(output_file))

                    print(f"  Saved {len(case_data)} DICOM slices for case {case_id} to {case_dir}")

            except Exception as e:
                print(f"Error saving case {case_id}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: save individual slices as numpy arrays in case subdirectory
                case_dir = output_path / case_id
                case_dir.mkdir(exist_ok=True)
                for item in case_data:
                    output_array = item['output'][0, 0]
                    filename = item['filename']
                    np.save(case_dir / f"output_{Path(filename).stem}.npy", output_array)

        print(f"Test outputs saved successfully to {output_path}!")