

## Getting Started
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your Hugging Face token** 
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```
3. **Run an experiment**
   ```bash
   # Residual-norm accuracy, all 16 models, Legacy Survey -> HSC
   CUDA_VISIBLE_DEVICES=0 python experiments/residual_experiment_SVR.py --survey legacy \
       --out results/legacy_hsc_residuals.csv

   # Conformal coverage, a couple of models, JWST -> HSC
   CUDA_VISIBLE_DEVICES=0 python experiments/conformal_experiment_SVR.py --survey jwst \
       --models dino_base,vit_large --alpha 0.10 \
       --out results/jwst_hsc_conformal.csv
   ```
   `--models` accepts a comma-separated subset, so a sweep across all 16
   models can be sharded across processes/machines; it defaults to all
   16 foundation models if omitted.

