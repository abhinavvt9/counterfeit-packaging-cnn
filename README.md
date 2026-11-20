# Counterfeit Medicine Packaging Detection Using a CNN Trained from Scratch

In this project we built a small CNN, trained completely from scratch (no pretrained weights), that looks at a photo of medicine packaging and says if it is **genuine** or **counterfeit**. We compared it with an SVM that uses handcrafted features. We also checked the dataset itself for shortcuts, and that turned out to be the most interesting part.

*M.Tech IT, AI-ML course project (MTITE102), NSUT New Delhi.*

<p align="center"><img src="docs/figures/pipeline.png" width="800" alt="Pipeline"></p>

## What we found

- Our CNN has only **101,793 parameters** and gets **89.9% accuracy, 96.0% ROC-AUC and 92.4% recall** on fakes.
- The SVM gets 92.3% accuracy, but the difference is **not significant** (McNemar p = 0.09). The CNN misses fewer fakes.
- Using global average pooling + batch norm made our first CNN **42× smaller** (4.29M → 0.10M parameters) with no drop in accuracy.
- **The dataset has a big problem:** using only file details (format, size, brightness), a simple classifier gets **100%**. An SVM that sees only the border of the image, with the packaging hidden, still gets **90.7%**. So high accuracy on this dataset does not really prove a model can detect fakes.

| Model (grouped 5-fold CV, %) | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Handcrafted features + SVM | 92.3 ± 2.9 | 91.2 ± 3.4 | 89.9 ± 3.8 | 97.8 ± 0.8 |
| **Our CNN (from scratch)** | 89.9 ± 3.7 | **92.4 ± 3.9** | 87.4 ± 4.2 | 96.0 ± 2.1 |

<p align="center"><img src="results/error_analysis/gradcam_examples.png" width="600" alt="Grad-CAM"></p>

The full write-up is in [`paper/paper.pdf`](paper/paper.pdf). All tables are in `results/tables/` and the result of every fold is in `results/runs/`.

## Dataset

We used [Fake vs Real Medicine Datasets Images](https://www.kaggle.com/datasets/surajkumarjha1/fake-vs-real-medicine-datasets-images) from Kaggle. It is not included here, so download it and extract it into a folder called `dataset/`.

The zip has 2,224 files, but 1,578 of them are exact copies. After removing those (MD5) and a few mixed near-duplicate groups (pHash), we are left with **636 images** (398 genuine, 238 fake).

## How to run

```bash
pip install -r requirements.txt

python run_all.py --data dataset      # runs everything (use a GPU for the CNN)
python demo/app.py                    # web demo, works with the saved model in models/
```

No GPU? Open `run_on_colab.ipynb` in Google Colab, pick a T4 GPU and run the cells from top to bottom.

You can also run only some steps, for example `--stages index audit splits cache svm report` (no PyTorch needed).

## Folders

```
preprocessing/   cleaning, duplicate removal, audit, 5-fold grouped splits
baseline/        handcrafted features + SVM
cnn/             our CNN, training, Grad-CAM, final model
experiments/     the 7 CNN experiments (E1-E7)
analysis/        metrics, tables, figures, error analysis, McNemar test
demo/            Gradio web demo
models/          trained CNN (final_cnn.pt)
results/         all numbers and figures
paper/           research paper (IEEE format) + LaTeX
```

## Team

- **Abhinav Shankar Tripathi** (2026PIT4212)
- **Saksham Bhatnagar** (2026PIT4210)
- **Supervisor:** Dr. Ankush Jain, Department of IT, NSUT

## License

MIT. The dataset belongs to its Kaggle uploader.
