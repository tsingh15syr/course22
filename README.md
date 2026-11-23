# Practical Deep Learning for Coders · 2022 notebooks

The numbered notebooks in `notebooks/` follow the lecture sequence, [clean/](notebooks/clean/) contains the short versions, and [slides/](slides/) contains the two slide notebooks. They share the `course22` kernel. Use the working copies alongside the original recordings. The original course materials and historical outputs are archived in `original/`.

Start with [lesson 00](notebooks/00-is-it-a-bird-creating-a-model-from-your-own-data.ipynb). The working copies open with empty outputs so you can see the results of your own run.

## Set up the course kernel

The lock targets **Windows x64, CPython 3.12.11 and CPU PyTorch**. It includes the notebook packages, JupyterLab, and a compatible torch 2.8.0 / torchvision 0.23.0 pair. Use PowerShell from the repository root. On this machine the Miniforge `python` command provides the required interpreter.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python -m pip install uv==0.9.18
.\.venv\Scripts\uv sync --locked --python .venv\Scripts\python.exe
.\.venv\Scripts\uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python -m ipykernel install --sys-prefix --name course22 --display-name "Python (course22)"
.\scripts\setup_graphviz.ps1
```

If this checkout already has the project `.venv` and uv, start at `uv sync`. The first sync downloads the CPU runtime and scientific packages. The lock includes exact transitive versions and wheel hashes. All package installs and the kernel registration use the project environment.

- **VS Code** — Open the repository folder with the Python and Jupyter extensions enabled. Open the notebook in `notebooks`, click **Select Kernel**, then select **Python (course22)** or the Python environment at `.venv\Scripts\python.exe`. The setup cell prints the interpreter path.
- **JupyterLab** — Run `.\.venv\Scripts\python -m jupyterlab .` from the repository root. Open lesson 00 and select **Python (course22)**.

Run the notebook in order. The live collection cell makes six searches and can try up to 1,200 image URLs. The training cell fetches the ImageNet ResNet-18 weights on first use (about 45 MB) and runs the lecture's fine-tuning schedule. To check the installation on generated images, run `.\.venv\Scripts\python scripts\validate_00.py --pretrained` from the repository root. This exercises prediction with the lecture's pretrained backbone and skips fine-tuning. Omit `--pretrained` for an offline check with randomly initialized weights. [Torchvision ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## Use local photos

If a search is rate-limited, a preview URL refuses downloads, or you want to reuse a collection, set `use_local_photos = True` near the beginning of Step 1. Place these files in the kernel's working directory. Running `%pwd` in a notebook cell shows the directory.

```text
bird.jpg                 a held-out bird for the preview and final prediction
forest.jpg               a held-out forest for the preview
bird_or_not/
    bird/                copies of distinct bird photos
    forest/              copies of distinct forest photos without birds
```

A completed live run already produces this layout. For a collection of your own, aim for at least 50 varied photos per class so the original batch size and validation split have enough examples. Keep both previews outside the training folders. Use working copies because the lesson resizes images in place and deletes unreadable ones. Rerun from the switch cell. The previews, resize, cleanup, split, training and prediction use the same lesson code paths with your files.

For a fully offline run, the pretrained weights must also be cached. On a connected machine, completing the learner-construction line once populates the torch cache. To move it, copy `resnet18-f37072fd.pth` from the source machine's `checkpoints` directory to the same directory on the destination. Run `from torch.hub import get_dir; print(get_dir())` in each environment to locate the parent directory.

See the [audit and changelog](COMPATIBILITY.md) and the [validation results](VALIDATION.md).

## Data and hardware for later lessons

Paths below are relative to the notebook kernel's working directory (`%pwd`). JupyterLab usually starts a kernel in the notebook's directory. To reuse data between a lecture and clean copy, run both kernels in the same data directory or make the named folder available to each.

| Notebook | Inputs and first run |
| --- | --- |
| 01, 03, 04 | The shell demo targets Windows. The timm CSVs for 03 ship under `data/timm-2022`. Run 03 from `notebooks/`, `notebooks/clean/`, `data/`, or the repository root. Widgets in 04 use the Jupyter frontend. |
| 02 | fastai downloads the Oxford-IIIT Pets archive on first use. For offline use, copy the extracted `oxford-iiit-pet` folder under fastai's data directory, or place the official archive in its archive cache. Inspect `fastai_cfg().path('data')` and `fastai_cfg().path('archive')` from `fastai.vision.all` for these locations. |
| 05–07 | Use the official Titanic `train.csv` and `test.csv` in `titanic/`, or supply `titanic.zip`. Otherwise the setup cell uses the Kaggle API. The portable Graphviz setup above supplies the tree renderer for 07 and the diagrams in the birds slides. |
| 08–10 | Extract the official Paddy Disease Classification competition archive into `paddy-disease-classification/`, including `train_images/`, `test_images/`, `train.csv` and `sample_submission.csv`. Otherwise fastkaggle retrieves the competition data. The original timm pretrained checkpoints can be large. Prepare the matching checkpoints on a connected machine for offline study. |
| Birds slides | Place `bird.jpg` and the two `bird_or_not/` class folders from lesson 00 in the working directory and select `use_local_photos = True` to reuse them. The later demonstrations use fastai's CAMVID_TINY, ADULT_SAMPLE and ML_SAMPLE archives and ResNet-34 weights. The same fastai and torch cache procedures apply. |
| NLP slides | Put the patent competition CSVs in `us-patent-phrase-to-phrase-matching/`, or provide its same-named ZIP or Kaggle access. The lecture uses Microsoft's DeBERTa-v3-small tokenizer and checkpoint. Both use the pinned Hub revision in the notebook. Hugging Face's cache can be copied from a connected machine. Set `HF_HUB_OFFLINE=1` for cache-only access. The housing example uses scikit-learn's `fetch_california_housing` cache. |

For live Kaggle downloads, accept the competition rules with your own account and configure your personal `~/.kaggle/kaggle.json` outside the notebook. The Paddy submission and publishing cells are separately opt-in. Publishing expects the notebook file in the kernel's current directory. [Titanic](https://www.kaggle.com/competitions/titanic/data) · [Paddy](https://www.kaggle.com/competitions/paddy-disease-classification/data) · [Patent phrase matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data)

The locked environment uses CPU float32. The conditional precision cells use FP16 on a CUDA-capable PyTorch installation. A CPU run of lesson 10 cannot reproduce its GPU-memory experiment. Runtime, roundoff and training results can differ from the lecture. Full training of the larger models can be impractical on CPU.

## Repository layout and shared data

Modern lecture notebooks are in `notebooks/`, practice copies in `notebooks/clean/`, and modern slide notebooks in `slides/`. Original notebooks, slide decks, spreadsheets, tools, CSS and historical container configuration are preserved together in `original/`. Validation and Graphviz setup utilities are in `scripts/`.

Use `data/` as the kernel working directory for shared datasets and downloaded images. In a newly opened lecture notebook run `%cd ../data`, in a clean notebook run `%cd ../../data`, and in a slide notebook run `%cd ../data` in a temporary cell. Check `%pwd` before changing directories. The lecture data paths then resolve within this shared folder. Fastai and Hugging Face retain their existing cache locations. For the optional Kaggle publishing demonstration, return to the directory containing the notebook before executing its publishing cell.
