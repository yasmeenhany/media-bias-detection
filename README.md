# Smaller Models, Cleaner Signals: Improving Sentence-Level Media Bias Detection on BABE

Code, data and results for our study of sentence-level media bias detection on the
[BABE](https://aclanthology.org/2021.findings-emnlp.101) dataset (Bias Annotations By Experts).
We build on Spinde et al. (2021) by evaluating newer transformer encoders (ERNIE 2.0, DeBERTa-v3, Flan-T5),
LLM- and paraphrase-based data augmentation, and distant pretraining on a small but cleanly labeled
corpus of political social media posts.

📄 **Report:** [paper/report.pdf](paper/report.pdf)

## Highlights

- **DeBERTa-v3-small reaches 0.812 macro F1** on both BABE subsets. The best prior results were
  0.778 on SG1 and 0.804 on SG2 (BERT + distant pretraining).
- **Smaller is better here.** On the 1,700-sentence SG1 subset, `deberta-v3-small` (0.810) clearly beats
  `base` (≤0.797) and `large` (≤0.685).
- **Label quality can matter more than corpus size for distant pretraining.** Pretraining on 5k
  crowd-annotated political posts helped more than pretraining on ~25× as many noisily labeled news headlines.
- **Augmentation helps on small data.** 200 extra sentences (Flan-T5 paraphrases at temperature 1.5, or
  GPT-4o generations from a few-shot prompt focused on nuanced bias) improve ERNIE and DeBERTa on SG1.
- **T5 works as an encoder.** Using T5 as a text-to-text classifier failed (0.51–0.53 F1). Using the Flan-T5
  encoder with a classification head reached 0.786 / 0.804.
- **Error analysis.** The models catch bias carried by explicitly loaded words but miss bias that depends
  on context (e.g., "Wuhan virus").

## Results

Macro F1 from stratified 5-fold cross-validation. Rows marked † are from Spinde et al. (2021).

| Model                                   | SG1 (1,700) | SG2 (3,700) |
|-----------------------------------------|:-----------:|:-----------:|
| Feature-based baseline †                | 0.680       | 0.720       |
| BERT †                                  | 0.750       | 0.790       |
| BERT + distant (news headlines) †       | 0.778       | 0.804       |
| RoBERTa + distant (news headlines) †    | 0.775       | 0.799       |
| ERNIE 2.0                               | 0.781       | 0.796       |
| T5 (encoder–decoder, text-to-text)      | 0.525       | 0.510       |
| Flan-T5 (encoder + classification head) | 0.786       | 0.804       |
| DeBERTa-v3                              | 0.806       | 0.810       |
| BERT + data augmentation                | 0.771       | –           |
| ERNIE + data augmentation               | 0.785       | –           |
| DeBERTa-v3 + data augmentation          | **0.812**   | –           |
| DeBERTa-v3 + distant (social media)     | **0.812**   | **0.812**   |
| DeBERTa-v3 + distant (news headlines)   | 0.810       | 0.804       |

See the report for hyperparameter sweeps (ERNIE, Flan-T5, DeBERTa) and the per-prompt augmentation results.

## Repository structure

```
├── notebooks/
│   ├── deberta_classification.ipynb   # DeBERTa-v3 5-fold CV + Optuna tuning
│   ├── ernie_classification.ipynb     # ERNIE 2.0 5-fold CV, with optional data augmentation
│   ├── t5_classification.ipynb        # T5 / Flan-T5: encoder–decoder and encoder + classifier head
│   ├── distant_pretraining.ipynb      # DeBERTa pretraining on news headlines or social media posts
│   ├── data_augmentation.ipynb        # T5 paraphrasing and GPT-4o prompt versions v1–v5
│   ├── results_analysis.ipynb         # Error analysis by opinion label (report Fig. 2)
│   ├── naive_bayes_baseline.ipynb     # Naive Bayes baseline
│   └── babe_reference/                # Original BABE pipeline from Spinde et al., kept for reference
│       ├── classification.ipynb             # BERT / RoBERTa / DistilBERT / ELECTRA / XLNet
│       ├── classification_baseline_model.ipynb  # XGBoost feature-based baseline
│       ├── features_engineering.ipynb       # Feature construction for the baseline
│       └── data_set_evaluation.ipynb        # Inter-annotator agreement and label statistics
├── data/
│   ├── final_labels_{MBIC,SG1,SG2}.*  # BABE aggregated labels (.csv and .xlsx)
│   ├── raw_labels_{MBIC,SG1,SG2}.*    # Individual annotator labels
│   ├── news_headlines_usa_*.csv       # Distantly labeled news headlines (Spinde et al.)
│   ├── dt_final_SG{1,2}.xlsx          # Engineered features for the baseline
│   ├── bias_word_lexicon.xlsx
│   └── augmentation/                  # Generated sentences (T5 paraphrases, GPT-4o prompts v1–v5)
├── results/                           # Per-fold ERNIE predictions for each augmentation setting
├── figures/                           # Training curves
├── docs/babe/                         # BABE annotation guidelines, questionnaire, original README
└── paper/report.pdf
```

## Setup

```bash
git clone https://github.com/yasmeenhany/media-bias-detection.git
cd media-bias-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks were developed on Google Colab with a GPU, and we recommend a GPU for all transformer
experiments. Each notebook begins with a setup cell that finds the repository root. To run on Colab,
clone the repo inside the notebook and `%cd` into it before running that cell.

### Social media data for distant pretraining

The [Political Social Media Posts](https://www.kaggle.com/datasets/crowdflower/political-social-media-posts)
dataset (CrowdFlower, 2016) is not redistributed here. Download it from Kaggle and save it as
`data/political_social_media.csv`.

## Reproducing the experiments

| Experiment                  | Notebook                                         | Notes                                                                                                             |
|-----------------------------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| DeBERTa                     | `notebooks/deberta_classification.ipynb`         | `microsoft/deberta-v3-small`, lr 1e-4, weight decay 0.01, batch 16, max length 150                                 |
| ERNIE (± augmentation)      | `notebooks/ernie_classification.ipynb`           | `nghuyong/ernie-2.0-base-en`; set `data_augmentation=` to choose an augmentation set. Results go to `results/`    |
| Flan-T5                     | `notebooks/t5_classification.ipynb`              | `google/flan-t5-base`; the encoder–decoder variant is kept (commented out) for reference                        |
| Data augmentation           | `notebooks/data_augmentation.ipynb`              | Paraphraser: `Vamsi/T5_Paraphrase_Paws`. GPT-4o outputs are already saved in `data/augmentation/`               |
| Distant pretraining         | `notebooks/distant_pretraining.ipynb`            | Run either the *Media Cloud* or the *Political Social Media Posts* section, then train; saves a `.pth` state dict |
| Error analysis              | `notebooks/results_analysis.ipynb`               | Reads the pickles in `results/`                                                                                   |

Augmented sentences are added to the training folds only, never to the validation folds.

> **Known gap:** the step that loads the distant-pretrained DeBERTa weights and runs 5-fold fine-tuning on
> BABE is not yet included in `deberta_classification.ipynb`.

Trained model weights are not included because of their size.

## Authors

Cole Ofsthun, HyunDuk Seo, Yasmeen Abdelhafeez. Final project for CS 7643 Deep Learning at Georgia Tech, Fall 2024.

- **HyunDuk Seo:** ERNIE, hyperparameter tuning, data augmentation, error analysis
- **Yasmeen Abdelhafeez:** T5 / Flan-T5 variants, social media distant pretraining
- **Cole Ofsthun:** DeBERTa, baseline models, dataset visualization

## Acknowledgements and citation

This project builds directly on the BABE dataset and code by the
[Media Bias Group](https://media-bias-research.org/). The `data/` files (except `data/augmentation/`) and
`notebooks/babe_reference/` come from their
[original repository](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE).
If you use the data, please cite their paper:

```bibtex
@inproceedings{spinde2021babe,
  title     = {Neural Media Bias Detection Using Distant Supervision With {BABE} - Bias Annotations By Experts},
  author    = {Spinde, Timo and Plank, Manuel and Krieger, Jan-David and Ruas, Terry and Gipp, Bela and Aizawa, Akiko},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages     = {1166--1177},
  year      = {2021},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2021.findings-emnlp.101}
}
```

## License

[AGPL-3.0](LICENSE), inherited from the original BABE repository.
