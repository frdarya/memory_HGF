# TAPAS Observation Models for Recognition Memory with Similar Foils

## Overview

This repository contains custom TAPAS observation models designed to explain cross-type transfer effects in recognition memory tasks with parametrically similar foils. The models were developed to provide mechanistic explanations for behavioral findings and generate trial-by-trial parameters for model-based fMRI analysis.

### Behavioral Task
- **Study phase**: Participants study target images (2 presentations each)
- **Test phase**: Old/new recognition judgments on:
  - Targets (studied items)
  - F1 foils (75% similar)
  - F2 foils (50% similar)  
  - F3 foils (25% similar)

### Key Behavioral Finding
**Cross-type transfer effect**: Errors on one item type improve performance on the complementary item type from the same set:
- Miss on Target → 16.5% better F1 rejection
- False alarm on F1 → 11.7% better Target detection
- Effects persist across 2-300 trial lags
- Effects are set-specific (no generalization across sets)

## Model Architecture

### Hierarchical Structure
```
Input (u) → [HGF Binary] → Beliefs (mu) → [Observation Model] → Response (y)
              ↑                              ↑
     tapas_hgf_binary_config           Custom observation models
```

### Core Innovation: Similarity-Based Evidence Scaling
All models implement cubic similarity scaling based on Stevens' Power Law:
```matlab
scaled_evidence = mu1 × similarity³
```

This creates a strong discrimination gradient:
- Target (100%): 1.0³ = 1.000
- F1 (75%): 0.75³ = 0.422
- F2 (50%): 0.50³ = 0.125
- F3 (25%): 0.25³ = 0.016

## Model Descriptions

### 1. Static Model (Baseline)
**File**: `tapas_similarity_sets_binary_static.m`

**Equation**: 
```
Evidence = mu1 × similarity³
P("old") = sigmoid(ze × evidence)
```

**Parameters**:
- `ze`: Decision noise (only parameter)

**Purpose**: Baseline model with no learning or adaptation

### 2. Interaction Model
**File**: `tapas_similarity_sets_interaction.m`

**Equation**:
```
Evidence = mu1 × similarity³ × (1 + β × normalized_set_id)
P("old") = sigmoid(ze × evidence)
```

**Parameters**:
- `ze`: Decision noise
- `β_interaction`: Similarity × set interaction

**Purpose**: Tests if some sets are inherently easier/harder

### 3. Weighted Exposure Model (Best Performance)
**File**: `tapas_similarity_exposure_precomputed.m`

**Equation**:
```
Evidence = mu1 × similarity³ × (1 + β × weighted_exposure)
P("old") = sigmoid(ze × evidence)
```

**Parameters**:
- `ze`: Decision noise
- `β_exposure`: Exposure interference weight

**Purpose**: Models cumulative interference from similar items

## Model Comparison Results

| Model | Mean AIC | Best Fit Count | Key Finding |
|-------|----------|----------------|-------------|
| Static | 410.59 | 9/26 (34.6%) | Baseline |
| Interaction | 410.84 | 5/26 (19.2%) | No consistent set effects |
| **Weighted Exposure** | **409.46** | **12/26 (46.2%)** | **β = -0.425, p < 0.0001** |

### Key Findings
1. **Weighted exposure model wins** with lowest AIC
2. **Negative β_exposure** indicates interference: Previous exposure to similar items impairs discrimination
3. **96.2% of subjects** show negative β (highly consistent)
4. **Similarity-specific interference**:
   - Strongest for high similarity (T/F1): r = -0.335
   - Weakest for low similarity (F2/F3): r = -0.166
   - Difference significant (z = -7.31, p < 0.0001)

## Implementation Details

### Input Matrix Format
All models expect input matrix `u` with columns:
1. `old_new`: Ground truth (1=old, 0=new)
2. `similarity`: Perceptual similarity (0.25, 0.5, 0.75, or 1.0)
3. `set_id`: Unique identifier for each set
4. `weighted_exposure`: Pre-computed cumulative similarity (exposure model only)

### Pre-computing Weighted Exposure
```matlab
function weighted_exposure = compute_exposure_weights(data)
    % For each trial, sum similarities of all previous items from same set
    for each set
        for each trial in set
            if first trial
                weighted_exposure = 0
            else
                weighted_exposure = sum(previous_similarities_in_set)
            end
        end
    end
end
```

### Parameter Priors
Based on empirical fits across 26 subjects:
```matlab
% Decision noise prior (all models)
c.logze_mu = log(5);    % Empirical range: 3-10
c.logze_sa = 1;         % Moderate variance

% Exposure weight prior (exposure model)
c.beta_exposure_mu = 0;  % No effect
c.beta_exposure_sa = 1;  % Allow negative (interference)
```

## Usage Example

```matlab
% Load data
data = readtable('recognition_data.csv');

% Prepare input for single subject
subj_data = data(data.subj == 1, :);
y = subj_data.Response;  % Binary responses

% For exposure model, pre-compute weights
weighted_exposure = compute_exposure_weights(subj_data);
u = [subj_data.Old_new, subj_data.Similarity, subj_data.SetID, weighted_exposure];

% Fit models
output_static = tapas_fitModel(y, u(:,1:3), ...
    'tapas_hgf_binary_config', 'tapas_similarity_sets_binary_static_config');

output_exposure = tapas_fitModel(y, u, ...
    'tapas_hgf_binary_config', 'tapas_similarity_exposure_precomputed_config');

% Compare models
fprintf('Static AIC: %.2f\n', output_static.optim.AIC);
fprintf('Exposure AIC: %.2f\n', output_exposure.optim.AIC);
fprintf('Exposure β: %.3f\n', output_exposure.p_obs.beta_exposure);
```

## Theoretical Interpretation

### Mechanism: Similarity-Weighted Interference
1. **Within-set interference**: Seeing similar items creates confusion
2. **Cumulative effect**: Interference accumulates with each exposure
3. **Similarity weighting**: More similar items create more interference
4. **Cross-type relevance**: Target-F1 confusion could trigger error-driven learning

### Connection to Behavioral Effects
The exposure model suggests a two-stage process:
1. **Interference phase**: Similar items create discrimination errors
2. **Learning phase**: Errors may trigger adaptive mechanisms (not modeled here)


## File Structure
```
├── tapas_similarity_sets_binary_static.m          # Baseline model
├── tapas_similarity_sets_binary_static_config.m   # Baseline config
├── tapas_similarity_sets_binary_static_transp.m   # Baseline transform
├── tapas_similarity_sets_interaction.m            # Interaction model
├── tapas_similarity_sets_interaction_config.m     # Interaction config
├── tapas_similarity_sets_interaction_transp.m     # Interaction transform
├── tapas_similarity_exposure_precomputed.m        # Exposure model
├── tapas_similarity_exposure_precomputed_config.m # Exposure config
├── tapas_similarity_exposure_precomputed_transp.m # Exposure transform
├── compute_exposure_weights.m                     # Helper function
├── analyze_exposure_by_similarity.m               # Analysis script
└── README.md                                      # This file
```

## Requirements
- MATLAB R2019b or later
- TAPAS toolbox (https://www.translationalneuromodeling.org/tapas)
- Statistics and Machine Learning Toolbox
