TSRAM-Net
Reference-anchored sparse retrieval for open-library Raman unmixing.
TSRAM-Net identifies the one to three phases present in an unknown Raman
spectrum and estimates their fractions against an open library of thousands of
reference spectra, in a single forward pass on one CPU thread. The trainable
front end is decoupled from a frozen convex solver, so the same analysis layer
ports to a new instrument without re-deriving the solver.
Code accompanying the manuscript submitted to IEEE Journal of Selected Topics
in Quantum Electronics, Special Issue on Advances in Multifunctional Optical
Spectrometers.
Manuscript
Computational Open-Library Raman Spectrometry for Handheld and Planetary
Instruments: Coarse-to-Fine Retrieval With a Frozen Physical Refinement
Stage
Zhen Wang, Zhiqiang Xu, Simiao Ba, Tiantian Gao, Yutianxin Wang, Di Gao,
Jiajia He, Shaohua Chen, Taotao Mou
Beijing Information Science and Technology University, Beijing 100192, China
What is in this repository
The root holds two files:
README.md              # this file
TSRAM-Net-lite.zip     # the code, data and result files (5.2 MB)
Download TSRAM-Net-lite.zip and unzip it. It expands to:
TSRAM-Net-lite/
├── code/            # 25 scripts; one per experiment, plus the plotting code
├── data/
│   ├── md5/         # 177 measured binary mixtures (Carey et al., 2016)
│   ├── qi/          # ternary set + Chang'e-5 lunar soil (Qi et al., 2023)
│   └── rruff/       # empty; you supply the RRUFF library (see below)
├── checkpoints/     # empty in this archive; see "Trained weights"
├── results/         # 13 JSON files with the numbers behind every figure
├── README.md        # detailed documentation, script-to-figure map
├── requirements.txt
└── LICENSE
The README inside the archive is the full documentation. This page is only the
entry point.
Quick start
# 1. get the code
git clone https://github.com/Coolie702/TSRAM_Net.git
cd TSRAM_Net
unzip TSRAM-Net-lite.zip
cd TSRAM-Net-lite

# 2. install
pip install -r requirements.txt

# 3. supply the RRUFF library (required for anything at K > 20)
#    download the *Processed* 532 nm spectra from https://rruff.info/zipped_data_files/raman/
#    and unzip them flat into data/rruff/

# 4. redraw a figure from the shipped result files -- no training needed
cd code
python plot_kscale_latency.py
Everything under results/ was produced by the scripts in code/ and is
included so that the figures can be regenerated without retraining anything.
Headline numbers
Setting
Result
Latency, one CPU thread, batch 1
3.9 ms at $K=1900$; 11.4 ms at $K=10^4$ (~1 us per library entry)
Abundance cosine at $K=1900$
0.969 (dense MLP head: 0.886)
Presence F1, RRUFF-1000, 1% noise
0.79
Presence F1, MD5 177 real mixtures
0.793 (SUnSAL 0.681, FCLS 0.620)
20% noise, abundance cosine
0.442 (SUnSAL 0.304, FCLS 0.278)
Script-to-figure map
Manuscript element
Script(s)
Fig. 1, architecture
gen_architecture_fig_v9.py
Table I, RRUFF-1000 at 1% noise
highnoise_arch.py + eval_noise_dense_sameproto.py
Fig. 2, library-size scaling and latency
kscale_experiment_v4.py, baseline_sunsal_kscale.py, bench_latency_kscale.py, plot_kscale_latency.py
Fig. 3, noise robustness
eval_noise_sunsal.py, plot_figs_final.py
Fig. 4, instrument error budget
train_drift_full.py, diag_resolution.py, diag_resolution_deconv.py, eval_empirical_drift.py, plot_instrument_budget_final.py
Table II, MD5 177 mixtures
train_md5_v4_bg.py, eval_md5_authoritative_v4bg.py
Sec. IV-H, ternary and lunar soil
qi_step2_sunsal.py, qi_step3_tsram.py
Supp. S-XIII, contrastive vs BCE loss
train_tsram_bce.py
Supp. S-XV, presence-branch diagnosis
diag_pres_head.py, diag_pres_head2.py
Supp., weak-component reconstruction
plot_weakmin_final.py
Supp., component ablation
ablation_full_v3.py
Trained weights
This archive is code-only. The five checkpoints (121 MB total) are omitted to
keep it inside the 25 MB per-file limit of the submission system, and are
deposited with the archival release of the article.
Training reproduces them:
cd code
python highnoise_arch.py 1000              # all four Table I models, K = 1000
mv _ckpt_*_K1000.pt ../checkpoints/        # the scripts look for them there
Four models, several hours on one GPU. The protocol is identical across models:
same library, same 50k/10k split, same noise augmentation, AdamW at 8e-4,
600-epoch cap, early stopping on a fixed 1%-noise validation set.
Because RRUFF is a live database, retraining today will not reproduce the exact
numbers in the manuscript: the set of minerals ranked into the top $K$ shifts as
the database grows. The shipped results/ files are the reference values.
Reference library
The RRUFF spectra are not redistributed here (567 MB, and the RRUFF terms
ask that users obtain them from the source). Download the Processed 532 nm
spectra from
https://rruff.info/zipped_data_files/raman/
and unzip them flat into data/rruff/. At least 1,900 distinct minerals are
needed to reproduce $K=1900$; a smaller download still runs, at a
correspondingly smaller $K$.
Please cite the RRUFF Project:
Lafuente B, Downs RT, Yang H, Stone N (2015) The power of databases: the RRUFF
project. In: Highlights in Mineralogical Crystallography, 1--30.
A note on code style
This is research code, published as it was run. Two conventions are worth
knowing before reading it:
1. Scripts reuse each other by exec. Several scripts load the first part of
a parent script to inherit the library builder, the mixture generator, the
model classes and the evaluation functions, rather than importing them as a
module. The exec boundary is marked by a comment in each file. This keeps
every experiment byte-identical to the protocol it was compared against.
2. Paths are relative to the script, resolved through
os.path.dirname(os.path.abspath(__file__)). Scripts run from any working
directory, but result JSONs are written to the current directory.
We deliberately did not refactor these scripts into a package: the numbers in
the manuscript were produced by exactly these files.
Citation
@article{wang2026tsram,
  title   = {Computational Open-Library Raman Spectrometry for Handheld and
             Planetary Instruments: Coarse-to-Fine Retrieval With a Frozen
             Physical Refinement Stage},
  author  = {Wang, Zhen and Xu, Zhiqiang and Ba, Simiao and Gao, Tiantian and
             Wang, Yutianxin and Gao, Di and He, Jiajia and Chen, Shaohua and
             Mou, Taotao},
  journal = {IEEE Journal of Selected Topics in Quantum Electronics},
  year    = {2026},
  note    = {Submitted}
}
License
MIT, for the code. The RRUFF spectra are distributed under the RRUFF Project
terms and are not covered by this license.
