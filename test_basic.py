#!/usr/bin/env python3
"""
Basic smoke test to verify the pipeline components work.
Run this after installing dependencies to catch obvious issues.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not installed - run: pip install torch torchvision")
        return False
    
    try:
        from src.models.classification_2d import create_model, EfficientNetB0Classifier
        print("✓ 2D classification models")
    except Exception as e:
        print(f"✗ 2D models failed: {e}")
        return False
    
    try:
        from src.models.classification_3d import create_3d_model
        print("✓ 3D classification models")
    except Exception as e:
        print(f"✗ 3D models failed: {e}")
        return False
    
    try:
        from src.models.segmentation import create_segmentation_model
        print("✓ Segmentation models")
    except Exception as e:
        print(f"✗ Segmentation models failed: {e}")
        return False
    
    try:
        from src.data.dataset import MRIDataset, create_dataloaders
        print("✓ Dataset components")
    except Exception as e:
        print(f"✗ Dataset failed: {e}")
        return False
    
    try:
        from src.losses.losses import create_loss_function
        print("✓ Loss functions")
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that models can be instantiated."""
    print("\nTesting model creation...")
    
    import torch
    
    try:
        from src.models.classification_2d import create_model
        
        # Test EfficientNet
        model = create_model('efficientnet_b0', num_classes=4)
        print(f"✓ EfficientNet-B0 created: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Test ResNet
        model = create_model('resnet50', num_classes=4)
        print(f"✓ ResNet50 created: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass works: {output.shape}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    return True


def test_data_pipeline():
    """Test data loading with dummy data."""
    print("\nTesting data pipeline...")
    
    try:
        from src.data.dataset import get_transforms
        import torch
        import numpy as np
        
        # Test transforms
        transform = get_transforms('train', image_size=(224, 224))
        print("✓ Training transforms created")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = transform(image=dummy_image)
        print(f"✓ Transform works: {transformed['image'].shape}")
        
    except Exception as e:
        print(f"✗ Data pipeline failed: {e}")
        return False
    
    return True


def test_loss_functions():
    """Test loss function creation."""
    print("\nTesting loss functions...")
    
    try:
        from src.losses.losses import create_loss_function
        import torch
        
        # Test different loss functions
        losses = ['cross_entropy', 'weighted_ce', 'focal', 'dice']
        
        for loss_name in losses:
            try:
                loss_fn = create_loss_function(loss_name, num_classes=4)
                
                # Test forward pass
                logits = torch.randn(4, 4)  # batch_size=4, num_classes=4
                targets = torch.randint(0, 4, (4,))
                
                loss_value = loss_fn(logits, targets)
                print(f"✓ {loss_name}: {loss_value.item():.4f}")
                
            except Exception as e:
                print(f"✗ {loss_name} failed: {e}")
                
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        return False
    
    return True


def test_training_components():
    """Test training utilities."""
    print("\nTesting training components...")
    
    try:
        from src.utils.training import create_optimizer, create_scheduler
        from src.models.classification_2d import create_model
        import torch
        
        # Create dummy model
        model = create_model('efficientnet_b0', num_classes=4)
        
        # Test optimizer creation
        optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3)
        print(f"✓ AdamW optimizer created")
        
        # Test scheduler creation
        scheduler = create_scheduler(optimizer, 'cosine', num_epochs=100)
        print(f"✓ Cosine scheduler created")
        
    except Exception as e:
        print(f"✗ Training components failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Running basic smoke tests...\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_pipeline, 
        test_loss_functions,
        test_training_components
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All basic tests passed! The pipeline should work.")
    else:
        print("⚠️  Some tests failed. Check dependencies and fix issues before training.")
    
    print("\nTo install dependencies:")
    print("pip install -r requirements.txt")


if __name__ == '__main__':
    main()