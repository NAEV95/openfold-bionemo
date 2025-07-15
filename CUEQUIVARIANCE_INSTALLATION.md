# cuEquivariance Installation Guide

This guide explains how to install cuEquivariance support for OpenFold-BioNeMo to enable accelerated triangle attention and multiplicative update operations.

## Overview

cuEquivariance is an optional dependency that provides GPU-accelerated kernels for triangle operations in OpenFold-BioNeMo. When enabled, it can significantly speed up inference and training by using optimized CUDA kernels.

## Requirements

- **CUDA**: cuEquivariance requires CUDA-compatible GPU hardware
- **Linux**: Currently only supported on Linux platforms (not macOS)
- **Python**: 3.9 or later
- **PyTorch**: Compatible with PyTorch 1.12+
- **Triton**: Version 3.3.0 or later (automatically installed with cuEquivariance)

## Installation Methods

### Method 1: Using pip with optional dependencies

```bash
# Install OpenFold with cuEquivariance support
pip install -e .[cuequivariance]
```

### Method 2: Using requirements file

```bash
# Install cuEquivariance dependencies separately
pip install -r requirements-cuequivariance.txt
```

### Method 3: Using conda environment

```bash
# Create environment with cuEquivariance support
conda env create -f environment-cuequivariance.yml
conda activate openfold-cuequivariance-env
```

### Method 4: Manual installation

```bash
# Install cuEquivariance manually
pip install cuequivariance-torch
pip install triton>=3.3.0
```

## Enabling cuEquivariance

After installation, enable cuEquivariance in your configuration:

```python
# In your config or script
config.globals.use_cuequivariance = True
```

Or via command line:
```bash
python run_pretrained_openfold.py --use_cuequivariance
```

## Verification

To verify that cuEquivariance is properly installed and working:

```python
import torch
import importlib.util

# Check if cuEquivariance is available
cuequivariance_available = importlib.util.find_spec("cuequivariance_torch") is not None
print(f"cuEquivariance available: {cuequivariance_available}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuequivariance_available and cuda_available:
    # Test basic import
    from cuequivariance_torch.primitives.triangle import triangle_attention
    print("✓ cuEquivariance is ready to use!")
else:
    print("⚠️  cuEquivariance requires CUDA and proper installation")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cuequivariance_torch'**
   - Solution: Install cuEquivariance using one of the methods above

2. **ValueError: cuequivariance_torch not installed**
   - Solution: Install cuEquivariance dependencies and ensure they're in your Python path

3. **Triton version compatibility issues**
   - Solution: Install Triton 3.3.0 specifically: `pip install triton==3.3.0`

4. **CUDA compatibility issues**
   - Solution: Ensure your CUDA version is compatible with cuEquivariance requirements

5. **Performance not improving**
   - Verify that `config.globals.use_cuequivariance = True` is set
   - Check that you're running on GPU (not CPU)
   - Ensure your model size is large enough to benefit from cuEquivariance

### Configuration Validation

The system automatically validates cuEquivariance configuration:

- **Dependency Check**: Verifies cuEquivariance is installed when enabled
- **Mutual Exclusivity**: Ensures cuEquivariance doesn't conflict with other attention mechanisms
- **Runtime Validation**: Gracefully falls back to standard kernels if cuEquivariance fails

## Performance Considerations

- **Best Performance**: Large models with long sequences benefit most from cuEquivariance
- **Memory Usage**: cuEquivariance may use more GPU memory than standard kernels
- **Compatibility**: Works with existing OpenFold-BioNeMo training and inference pipelines

## Fallback Behavior

If cuEquivariance is not available or fails:
- The system automatically falls back to standard triangle operations
- No functionality is lost, only performance optimization
- Warnings are logged when fallback occurs

## Support

For cuEquivariance-specific issues:
- Check the [cuEquivariance repository](https://github.com/cuequivariance/cuequivariance) for documentation
- Ensure your CUDA and PyTorch versions are compatible
- Verify that Triton 3.3.0 is properly installed

For OpenFold-BioNeMo integration issues:
- Check the configuration validation messages
- Verify that the global flag is properly set
- Test with standard kernels first to isolate cuEquivariance issues