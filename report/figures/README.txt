Place the following images in this directory for the LaTeX report:

1. baseline.png      - Generated image with baseline (no DQAR)
2. layers_33.png     - Generated image with 33% layer fraction, 20% warmup
3. layers_50.png     - Generated image with 50% layer fraction, 20% warmup (recommended)
4. layers_100.png    - Generated image with 100% layer fraction, 20% warmup

All images should be generated with the same seed (42) and class label for fair comparison.

To generate these images, run:
  python scripts/benchmark_config.py --layer-fraction 0.0 --save-images   # baseline
  python scripts/benchmark_config.py --layer-fraction 0.33 --save-images
  python scripts/benchmark_config.py --layer-fraction 0.50 --save-images
  python scripts/benchmark_config.py --layer-fraction 1.0 --save-images
