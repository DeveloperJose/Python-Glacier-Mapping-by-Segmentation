#!/usr/bin/env python3
"""Simple test script for Lightning migration."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all imports work."""
    print("Testing imports...")
    
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Lightning: {e}")
        return False
    
    try:
        import mlflow
        print(f"✓ MLflow {mlflow.__version__}")
    except ImportError as e:
        print(f"✗ MLflow: {e}")
        return False
    
    try:
        import torchmetrics
        print(f"✓ TorchMetrics {torchmetrics.__version__}")
    except ImportError as e:
        print(f"✗ TorchMetrics: {e}")
        return False
    
    try:
        from glacier_mapping.lightning import GlacierDataModule, GlacierSegmentationModule
        print("✓ Glacier Lightning modules")
    except ImportError as e:
        print(f"✗ Glacier Lightning modules: {e}")
        return False
    
    return True


def test_config_loading():
    """Test loading existing config."""
    print("\nTesting config loading...")
    
    try:
        import yaml
        config_path = project_root / "conf" / "unet_train.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded config from {config_path}")
        print(f"  - Dataset: {config.get('training_opts', {}).get('dataset_name', 'Unknown')}")
        print(f"  - Run name: {config.get('training_opts', {}).get('run_name', 'Unknown')}")
        print(f"  - Epochs: {config.get('training_opts', {}).get('epochs', 'Unknown')}")
        print(f"  - Batch size: {config.get('loader_opts', {}).get('batch_size', 'Unknown')}")
        print(f"  - Output classes: {config.get('loader_opts', {}).get('output_classes', 'Unknown')}")
        
        return config
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return None


def test_data_module(config):
    """Test creating data module."""
    print("\nTesting data module...")
    
    try:
        from glacier_mapping.lightning import GlacierDataModule
        
        # Use a dummy path for testing
        datamodule = GlacierDataModule(
            processed_dir="/tmp/test_data",  # Dummy path
            batch_size=2,
            use_channels=[0, 1, 2],
            output_classes=[1, 2],
            class_names=["BG", "CleanIce", "Debris"],
            normalize="mean-std",
        )
        
        print("✓ GlacierDataModule created successfully")
        print(f"  - Batch size: {datamodule.batch_size}")
        print(f"  - Channels: {datamodule.use_channels}")
        print(f"  - Classes: {datamodule.output_classes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data module creation failed: {e}")
        return False


def test_model_creation(config):
    """Test creating model."""
    print("\nTesting model creation...")
    
    try:
        from glacier_mapping.lightning import GlacierSegmentationModule
        
        model = GlacierSegmentationModule(
            model_opts=config.get('model_opts', {}),
            loss_opts=config.get('loss_opts', {}),
            optim_opts=config.get('optim_opts', {}),
            scheduler_opts=config.get('scheduler_opts', {}),
            class_names=["BG", "CleanIce", "Debris"],
            output_classes=[1, 2],
            use_channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        
        print("✓ GlacierSegmentationModule created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_trainer_creation():
    """Test creating trainer."""
    print("\nTesting trainer creation...")
    
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        
        trainer = pl.Trainer(
            accelerator="cpu",  # Use CPU for testing
            devices=1,
            max_epochs=1,
            fast_dev_run=True,  # Just test setup
            logger=False,
            enable_checkpointing=False,
        )
        
        print("✓ Trainer created successfully")
        print(f"  - Accelerator: {trainer.accelerator}")
        print(f"  - Devices: {trainer.devices}")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Lightning Migration Test Suite ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return False
    
    # Test config loading
    config = test_config_loading()
    if config is None:
        print("\n❌ Config loading failed. Cannot continue.")
        return False
    
    # Test data module
    if not test_data_module(config):
        print("\n❌ Data module test failed.")
        return False
    
    # Test model creation
    if not test_model_creation(config):
        print("\n❌ Model creation test failed.")
        return False
    
    # Test trainer creation
    if not test_trainer_creation():
        print("\n❌ Trainer creation test failed.")
        return False
    
    print("\n✅ All tests passed! Lightning migration is ready.")
    print("\nNext steps:")
    print("1. Start MLflow server: ./scripts/launch_mlflow.sh")
    print("2. Run training: uv run python scripts/train_lightning.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)