"""Exercise the lesson's local-photo path in a temporary Jupyter kernel run."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use the lesson's ImageNet weights (downloads once if needed).")
    args = parser.parse_args()
    notebook = nbformat.read(next(Path(__file__).parent.glob("00-*.ipynb")), as_version=4)
    nbformat.validate(notebook)
    cells = notebook.cells
    assert cells[7].source.endswith("use_local_photos = False")
    assert cells[24].source.splitlines() == [
        "learn = vision_learner(dls, resnet18, metrics=error_rate)",
        "learn.fine_tune(3)",
    ]

    # These substitutions affect only the in-memory smoke test.
    cells[7].source = "use_local_photos = True"
    if args.pretrained:
        cells[24].source = cells[24].source.splitlines()[0]
        cells.append(nbformat.v4.new_code_cell(
            "from torchvision.models import ResNet18_Weights\n"
            "weights = ResNet18_Weights.DEFAULT.get_state_dict(progress=False, check_hash=True)\n"
            "assert torch.equal(learn.model[0][0].weight.detach().cpu(), weights['conv1.weight'])\n"
            "assert any(isinstance(t, Normalize) for t in dls.after_batch.fs)"
        ))
    else:
        cells[24].source = "learn = vision_learner(dls, resnet18, metrics=error_rate, pretrained=False)"
    cells.append(nbformat.v4.new_code_cell(
        "assert len(failed) == 1 and not Path(failed[0]).exists()\n"
        "assert Image.open(path/'bird'/'0.jpg').size == (400, 200)\n"
        "assert list(dls.vocab) == ['bird', 'forest']\n"
        "assert (len(dls.train_ds), len(dls.valid_ds)) == (64, 16)\n"
        "xb, yb = dls.one_batch()\n"
        "assert tuple(xb.shape[1:]) == (3, 192, 192) and len(xb) == len(yb)\n"
        "assert is_bird in dls.vocab and len(probs) == 2\n"
        "assert abs(float(probs.sum()) - 1) < 1e-5"
    ))

    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        for label, red in [("bird", 60), ("forest", 120)]:
            folder = work / "bird_or_not" / label
            folder.mkdir(parents=True)
            Image.new("RGB", (30, 20), (red, 90, 30)).save(work / f"{label}.jpg")
            for i in range(40):
                Image.new("RGB", (480, 240), (red, i, 20)).save(folder / f"{i}.jpg")
        (work / "bird_or_not" / "bird" / "invalid.jpg").write_bytes(b"invalid image")
        NotebookClient(notebook, kernel_name="course22", timeout=180).execute(cwd=tmp)
    print("Passed local previews, resize, cleanup, split, batch, ResNet-18 construction and prediction.")


if __name__ == "__main__":
    main()
