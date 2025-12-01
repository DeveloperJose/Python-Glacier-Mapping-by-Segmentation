#!/usr/bin/env python3
"""Simple Lightning training test."""

import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_setup():
    """Test basic PyTorch Lightning setup."""
    print("=== Basic Lightning Test ===\n")
    
    # Test imports
    try:
        import pytorch_lightning as pl
        import torch
        import mlflow
        import torchmetrics
        print(f"✓ All packages imported successfully")
        print(f"  - PyTorch Lightning {pl.__version__}")
        print(f"  - PyTorch {torch.__version__}")
        print(f"  - MLflow {mlflow.__version__}")
        print(f"  - TorchMetrics {torchmetrics.__version__}")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test config loading
    try:
        config_path = project_root / "conf" / "unet_train.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded from {config_path}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Test simple Lightning module
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        
        class SimpleModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.layer(x)
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                return loss
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())
        
        model = SimpleModule()
        print("✓ Simple Lightning module created")
        
    except Exception as e:
        print(f"✗ Lightning module creation failed: {e}")
        return False
    
    # Test trainer creation
    try:
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
        )
        print("✓ Lightning trainer created")
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        return False
    
    print("\n✅ Basic Lightning setup works!")
    return True


def test_original_framework():
    """Test original framework import."""
    print("\n=== Testing Original Framework ===\n")
    
    try:
        from glacier_mapping.core.frame import Framework
        print("✓ Original Framework imported")
        
        # Test loading from config
        config_path = project_root / "conf" / "unet_train.yaml"
        framework = Framework.from_config(str(config_path))
        print("✓ Framework loaded from config")
        print(f"  - Model parameters: {sum(p.numel() for p in framework.model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Framework test failed: {e}")
        return False


def main():
    """Run tests."""
    print("Lightning Migration Validation\n")
    
    # Test basic setup
    if not test_basic_setup():
        print("\n❌ Basic Lightning setup failed")
        return False
    
    # Test original framework
    if not test_original_framework():
        print("\n❌ Original framework test failed")
        return False
    
    print("\n✅ All tests passed!")
    print("\nNext steps:")
    print("1. The basic Lightning setup works")
    print("2. Original Framework is functional")
    print("3. We can proceed with gradual migration")
    print("4. Start MLflow server when ready: ./scripts/launch_mlflow.sh")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)