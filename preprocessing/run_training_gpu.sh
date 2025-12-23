#!/bin/bash
# Script to run training with GPU support
# Sets up proper CUDA library paths for TensorFlow

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vsl

# Set CUDA library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Print environment info
echo "========================================"
echo "Environment Setup"
echo "========================================"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check GPU
echo "========================================"
echo "GPU Detection"
echo "========================================"
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs found: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]"
echo ""

# Run training
echo "========================================"
echo "Starting Training"
echo "========================================"
python -m training.run
