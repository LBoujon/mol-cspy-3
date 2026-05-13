# Faraday2024 MACE Potential Models

This directory contains trained MACE potential models for the Faraday2024 paper:
"Predictive crystallography at scale: mapping, validating, and learning from 1000 crystal energy landscapes"
https://doi.org/10.1039/D4FD00105B

## Available Models

- **Faraday2024_stage1_mace.model**  
    `rigid_csp_mlp_iteration4_rattled_mlp_opt_pbed3_small.model`

- **Faraday2024_stage2_mace.model**  
    `rigid_csp_mlp_iteration4_rattled_mlp_opt_pbed3_small_swa.model`

## Important Note

These models are trained on PBE-D3 total energies and forces. **Do not add D3 corrections to the predictions** when using these models.
