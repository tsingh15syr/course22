# Notebook validation

Tested on **Windows x64, Python 3.12.11, CPU** in the project-local `course22` Jupyter kernel. The 23 additional notebooks comprise ten lecture copies, eleven clean copies and two slide decks. Lesson 00's previously validated lecture copy is reported separately below.

**Fully validated** means every executable lesson cell ran as written on its intended input. **Partially validated** means the smoke test passed, but used fixtures, shortened a fit, or omitted an external or hardware-dependent step. **Blocked** means a notebook could not complete its representative smoke path. Browser gestures and visual fidelity of interactive frontends were outside the kernel checks.

| Notebook copies | Status | What ran | Coverage limit |
| --- | --- | --- | --- |
| Clean 00 (1) | Partial | Generated bird/forest images, original split and transforms, cached ImageNet ResNet-18, both probability outputs. | Local-photo route, no fine-tuning. The reference lecture's live preview checks are recorded below. |
| 01 lecture + clean (2) | Full | Arithmetic, current-directory shell escape, variables, Matplotlib, IPython help and timing. | Manual editing and keyboard exercises are for the learner. |
| 02 lecture + clean (2) | Partial | Filename labels on 80 Pets-shaped fixture images, original loader and pretrained learner, pickle export, reload and prediction. | Oxford-IIIT Pets archive and fine-tuning omitted. |
| 03 lecture + clean (2) | Full | All original joins, filters, descriptive figures and Plotly OLS trend lines ran on the three real historical CSVs. | The plotted timings are those recorded on the RTX 3090. |
| 04 lecture + clean (2) | Full | All seeded synthetic-data cells, widget construction and initial callbacks, MAE, gradients, manual parameter updates and ReLU demonstrations. | Slider dragging was not automated. |
| 05 lecture + clean (2) | Partial | Titanic-shaped data, imputation, dummy variables, a 120 × 12 tensor, hand-coded gradient updates, sigmoid plot, linear/shallow/deep variants, evaluation and CSV export. | One test epoch per model invocation on generated rows. |
| 06 lecture + clean (2) | Partial | Original feature engineering, tabular transforms and learner, 24 two-class predictions, submission export, and the ensemble averaging expression. | LR finder and model fits omitted. The averaging test used duplicate fixture predictions. |
| 07 lecture + clean (2) | Partial | Split scoring, widgets, decision trees, Graphviz SVG, voting and forest predictions, feature importance and CSVs. | Synthetic Titanic rows and two trees for each ensemble in the test. |
| 08 lecture + clean (2) | Partial | Local Paddy layout, label folders, lecture augmentations, actual ResNet-26d model, test loader, decoded predictions and submission CSV. | Generated images, random model weights, CPU. LR search, training and remote actions omitted. |
| 09 lecture + clean (2) | Partial | Image resize, preprocessing variants, ResNet-26d forward, validation predictions, error rate, TTA and submission mapping. | Generated images and an untrained ResNet-26d. ConvNeXt training comparisons, competition data and remote actions omitted. |
| 10 lecture + clean (2) | Partial | Original one-class memory-probe data layout, ConvNeXt-small CPU forward pass, accumulation callback configured for 64 samples, registry weights for the ensemble architectures, duplicate weighting and CSV output. | No CUDA memory measurements, large-model weights or multi-model training. Eight fixture prediction tensors exercised the final ensemble. |
| Birds slide deck (1) | Partial | Bird classification with pretrained ResNet-18, Graphviz diagrams, CAMVID-shaped mask loader and ResNet-34 U-Net inference, tabular/collaborative learners, math widgets and polynomial plots. | Local fixture images and tables, randomly initialized U-Net, no fits or bulk searches. |
| NLP slide deck (1) | Partial | Patent-shaped CSVs, real pinned tokenizer, token IDs, seed-42 split, housing correlations on real scikit-learn data, and Trainer evaluation with finite Pearson correlation. Both recorded example token sequences matched exactly. | The smoke run used a small random DeBERTa regression model. Patent competition data, pretrained model weights, CUDA FP16 and training omitted. |

**Totals for this migration: 6 fully validated, 17 partially validated, 0 blocked.** Each entry covers a separate kernel execution for every named file. The original lesson 00 pretrained regression also passed after expanding the lock.

The expanded environment passed the offline lock and sync checks. Sync reported zero pending changes, and `uv pip check` found all installed packages compatible. All 24 original notebooks matched Git HEAD.

All 1,196 cells across the 24 working copies passed structural checks. Eight Markdown cells in the 23 new copies have updated text covering setup, data access or the local runtime. All other Markdown text in those copies matched its original. Only the kernel and language metadata changed at notebook level.

### Reproduce the checks

Use the [project setup](README.md), including the portable Graphviz renderer, and the reference ResNet-18 checkpoint. The offline course suite uses the real housing dataset and three small DeBERTa tokenizer/config files. To prepare these on a fresh machine, run the following once from the repository root.

```powershell
.\.venv\Scripts\python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"
.\.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/deberta-v3-small', revision='a36c739020e01763fe789b4b85e2df55d6180012', allow_patterns=['config.json','tokenizer_config.json','spm.model'])"
.\.venv\Scripts\python scripts\validate_course.py
```

[validate_course.py](scripts/validate_course.py) also accepts individual notebook paths. It compares each working copy with its original, validates the notebook schema, compiles IPython-transformed code, and checks cell count, order, types, IDs, metadata and attachments. It runs an in-memory copy in a temporary directory, limits CPU worker counts for fixtures, and keeps generated datasets, exports and submissions out of the teaching directories. Shortened fits, random weights and fixture substitutions in the table apply only to this test run.

The offline ZIP fallback passed separately for 05, 06, 07 and the patent slides. Kaggle client import and argument binding for download, submission and kernel publication were checked using dummy local configuration. Live authentication, competition rule acceptance, submission and publication remain unverified.

## Lesson 00 reference

**Lesson 00 is ready to follow on Windows x64 with the project kernel.** The notebook's local-photo path ran through prediction with the original pretrained learner. The test skipped the fine-tuning line in memory. The environment used CPython 3.12.11, fastai 2.8.12, torch 2.8.0+cpu and torchvision 0.23.0+cpu.

| Check | Result |
| --- | --- |
| Original notebooks | Compared all 24 originals with Git HEAD after normalizing checkout line endings. All matched. |
| Notebook structure | `nbformat.validate` passed. All 31 cells retained their type and position. All 12 code cells parsed. Cell metadata and attachments matched, and Pillow verified the embedded 606 × 489 comic. |
| Source comparison | Exactly nine cell sources changed. Eight handle project setup or image acquisition. The ninth updates the search provider's name in the overview. All source from “Step 2: Train our model” through the exercise matched the original. |
| Environment and lock | Installed the locked packages inside `.venv`. `uv lock --check --offline`, `uv sync --locked --check --offline` and `uv pip check` all passed. The sync check reported zero pending changes. |
| Jupyter | JupyterLab reported version 4.4.10. Started the project-local `course22` kernel with nbclient, ran the actual notebook through the first URL display, and asserted the kernel interpreter matched the project's Python. |
| Live search | The notebook helper returned a fastcore `L` with one URL for each of `bird photos` and `forest photos`. An earlier DDGS 9.9.3 trial failed while constructing its HTTP client. The locked DDGS 9.16.1 and primp 2.0.1 passed. |
| Live downloads | Re-executed the notebook through both preview cells in a temporary directory with fastdownload 0.0.8. The bird JPEG decoded at 1,631 × 1,800 and the forest JPEG at 5,472 × 3,078. These were separate live searches from the local-photo model test. |
| Image preparation and data loaders | Generated 40 RGB images per class, two separate previews and one invalid JPEG. The lesson displayed the previews, resized a 480 × 240 fixture to 400 × 200, removed the invalid file, and displayed a sample batch. The seeded split produced 64 training and 16 validation items. A training batch contained 3 × 192 × 192 tensors with corresponding labels and the vocabulary was `['bird', 'forest']`. |
| Pretrained ResNet-18 | The original `vision_learner(dls, resnet18, metrics=error_rate)` initialized on CPU. Verified the normalization transform and compared the first convolution weights exactly with the ImageNet checkpoint. The cached file's SHA-256 starts with `f37072fd`, matching the published weight filename. [Torchvision 0.23.0 source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/resnet.py#L292-L311) |
| Prediction | The original prediction cell returned a vocabulary label and two probabilities summing to one within 0.00001. The classifier head had received no fine-tuning, so the label carries no accuracy claim. |

### Reproduce the model check

After the [locked setup](README.md), run the pretrained smoke test from the repository root.

```powershell
.\.venv\Scripts\python scripts\validate_00.py --pretrained
```

The kernel operates in a temporary directory. The saved lesson retains the original fine-tuning call. Full training, validation accuracy, GPU timing and bulk image collection were outside this check.

### Issues encountered

The first dependency lock passed static checks but failed at runtime in fastai's data-loader code. Updating fastai and fastdownload resolved the labeling, transform recursion and missing padding-import errors. See the [compatibility changelog](COMPATIBILITY.md) for the affected versions. A Windows file-access error during removal of obsolete packages was resolved inside the project environment before the final sync. Jupyter emitted a Windows event-loop warning and completed the kernel run successfully.

## Filesystem cleanup validation

Relocated 81 tracked files using filesystem moves because the sandbox denied writes to the Git index. Verified all 24 original notebooks against their pre-move Git paths and all readable original materials against pre-move SHA-256 hashes. Verified 24 modernized notebooks against Git, including cell order, metadata and outputs. Only README links in both lesson 00 copies and benchmark lookup paths in both lesson 03 copies changed. All other modernized notebook contents matched.

Root documentation links resolve. Validation scripts parse and now locate original notebooks, working notebooks, slides and benchmark data in their new locations. All 183 checked environment fingerprints matched, covering installed package metadata, pyvenv.cfg, pyproject.toml and uv.lock. Notebook execution and runtime validation were outside this filesystem check. Prior runtime results above describe the modernization pass.

Removed the project-generated Python cache and retained the existing cache exclusions. The environment directory was preserved. The original slides/bird.jpg is a pre-existing self-referencing symbolic link whose target is bird.jpg. Its link target was preserved exactly in original/slides/bird.jpg. Original supporting PDF, PowerPoint and spreadsheet materials remain under original/ to retain their internal layout.
