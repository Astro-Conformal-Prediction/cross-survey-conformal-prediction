 Split conformal prediction for cross-survey embedding alignment
 
Experiment 3. Trains a k-NN translator from source-survey to target-survey
embeddings, then wraps its predictions in a distribution-free conformal region
and measures how well that region covers held-out objects.
 
## Install
 
```bash
pip install numpy pandas scikit-learn datasets
```
 
Python 3.9+. `numpy >= 1.22` is required — `compute_conformal_quantile` uses
`method="inverted_cdf"`, which was added in that release.
 
## Run
 
```bash
python run_experiment.py
```
 
Runs all three survey pairs × 16 backbones × 3 coverage levels and writes one
CSV per pair to `results/`.
 
Useful flags:
 
```bash
# one pair only
python run_experiment.py --pairs sdss_hsc
 
# a subset of backbones, for a quick smoke test
python run_experiment.py --pairs sdss_hsc --models astropt_15m dino_small
 
# different coverage levels
python run_experiment.py --coverages 0.9 0.99
 
# somewhere other than results/
python run_experiment.py --outdir runs/2026-07-31
```
 
Pair keys are `legacy_hsc`, `jwst_hsc`, `sdss_hsc`; the full Hugging Face
dataset paths they map to are in `PAIRS` at the top of `run_experiment.py`.
 
Start with a two-model smoke test. A full run is 16 backbones × 11 candidate
k values × 5 folds of brute-force k-NN per pair, which is hours on a laptop.
 
## Use as a library
 
```python
from pipeline import full_pipeline
 
df = full_pipeline("astro-conformal-project/split_sdss_hsc_embeddings")
df[df.target_coverage == 0.9].sort_values("radius")
```
