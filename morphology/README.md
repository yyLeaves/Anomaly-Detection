# Morphological Operations

## Quick Start

### 1. Update Configuration

Edit `config/morpho_validation.yaml` and set your data path:

```yaml
paths:
  output_base: "/path/to/your/postprocessed/prediction_masks"
```

This should point to your binary prediction masks directory (PNG format).

### 2. Run Tuning

```bash
python morphology/tune_morpho.py
```

The script will:
- Verify input data exists and is in correct format
- Process masks through all parameter combinations defined in config
- Generate comparison reports

## Configuration

`config/morpho_validation.yaml` contains tuning experiments with different parameter combinations:

```yaml
tuning_experiments:
  - name: "baseline"
    dilate_iterations: 1
    erode_iterations: 1
    num_rounds: 1
    
  - name: "aggressive"
    dilate_iterations: 2
    erode_iterations: 1
    num_rounds: 1
    # ... more combinations
```

To add new combinations, simply add another entry with different parameters.

## Core Processing

Core processing in `morphology/processor.py`:

1. **Closing Operation** (Dilation → Erosion)
   - Dilation: Expands anomaly regions, connects fragmented areas
   - Erosion: Restores approximate original size while maintaining connectivity
   - Repeated `num_rounds` times

2. **Connected Component Filtering**
   - Removes isolated components smaller than `min_component_size` pixels (default: 3)
   - Retains only meaningful connected anomalies

## Output

Results are saved to:
- `./output/validation_morpho_tuning/{experiment_name}/` - Processed masks
- `./reports/morphology_tuning/tuning_report.json` - Detailed statistics for each slice

## Understanding Results

**Aggressive parameters** (higher dilate_iterations):
- Preserve more area (higher % retained)
- Detect more anomalies (minimize false negatives)
- May include uncertain/borderline detections

**Conservative parameters** (higher erode_iterations):
- Preserve less area (lower % retained)
- Keep only high-confidence detections (minimize false positives)
- Remove edge regions and small uncertain areas

## Current Status

**Tuning Stage**: Parameter optimization on pred masks of **Fastflow**

![Comparison](../images/comparison.png)
![Summary](../images/table.png)


**Next Steps**:
1. Test on other models val data
2. Apply to test set with fixed parameters
3. Integration with volume processor
