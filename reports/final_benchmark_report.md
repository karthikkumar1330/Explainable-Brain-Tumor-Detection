# Three-Version Comparative Benchmarking Report
## 1. Classification Metrics Comparison (EfficientNet-B0)
| Metric | Version 1 (Untrained) | Version 2 (Baseline) | Version 3 (Fine-Tuned) |
|---|---|---|---|
| Accuracy | 0.2500 | 0.2500 | 0.5000 |
| Precision | 0.0625 | 0.3125 | 0.8333 |
| Recall | 0.2500 | 0.2500 | 0.5000 |
| F1-Score | 0.1000 | 0.1938 | 0.4917 |
| Loss | 1.3864 | 1.3311 | 1.2533 |
| Latency | 11.79ms | 10.52ms | 11.62ms |

## 2. Segmentation Metrics Comparison (UNeXt)
| Metric | Version 1 (Untrained) | Version 2 (Baseline) | Version 3 (Fine-Tuned) |
|---|---|---|---|
| IOU | 0.0000 | 0.0000 | 0.0000 |
| DICE | 0.0000 | 0.0000 | 0.0000 |
| Loss | 1.2711 | 20.6804 | 1.3689 |
| Latency | 11.78ms | 92.88ms | 136.20ms |

## 3. Visual Performance Chart
### Classification F1-Score Performance Comparison:
Version 1 (Untrained)  : ██ (10.0%)
Version 2 (Baseline)   : ███ (19.4%)
Version 3 (Fine-Tuned)  : █████████ (49.2%)

### Segmentation Dice Coefficient Performance Comparison:
Version 1 (Untrained)  :  (0.0%)
Version 2 (Baseline)   :  (0.0%)
Version 3 (Fine-Tuned)  :  (0.0%)
