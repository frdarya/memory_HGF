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
