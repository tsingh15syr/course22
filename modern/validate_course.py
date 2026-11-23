"""Run the remaining course notebooks in fresh course22 kernels on small local inputs."""
import argparse
import ast
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager
from nbclient import NotebookClient
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def image_files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (48, 32), (35, i % 255, 120)).save(folder / name)


def fixtures(lesson, work):
    if lesson in ("00", "birds"):
        for label in ("bird", "forest"):
            image_files(work / "bird_or_not" / label, [f"{i}.jpg" for i in range(40)])
        image_files(work, ["bird.jpg", "forest.jpg"])
    if lesson == "02":
        image_files(work / "pets/images", [f"{'Cat' if i % 2 else 'dog'}_{i}.jpg" for i in range(80)])
    if lesson == "birds":
        path = work / "camvid"
        image_files(path/"images", [f"{i}.png" for i in range(16)])
        (path/"labels").mkdir()
        for i in range(16):
            Image.new("L", (48, 32), i % 2).save(path/"labels"/f"{i}_P.png")
        (path/"codes.txt").write_text("background\nroad\n")
        adult = pd.DataFrame({"age":np.arange(100)+20, "fnlwgt":np.arange(100)+100,
            "education-num":np.arange(100)%10, "salary":["low","high"]*50})
        for name in ["workclass","education","marital-status","occupation","relationship","race"]:
            adult[name] = ["a","b"]*50
        adult.to_csv(work/"adult.csv",index=False)
        pd.DataFrame({"user":np.arange(100)%8,"movie":np.arange(100)%7,
                      "rating":np.arange(100)%5+1.0}).to_csv(work/"ratings.csv",index=False)
    if lesson == "03":
        shutil.copytree(HERE / "data", work / "data")
    if lesson in ("05", "06", "07"):
        path = work / "titanic"
        path.mkdir()
        i = np.arange(120)
        df = pd.DataFrame(dict(PassengerId=i+1, Survived=i%2, Pclass=i%3+1,
            Name=[f"Family{j}, Mr. Example" for j in i], Sex=np.where(i%3,"male","female"),
            Age=18.0+i%50, SibSp=i%4, Parch=i%3, Fare=5.0+i/3, Ticket=i//2,
            Cabin=np.where(i%3,"A11","D20"), Embarked=np.take(["C","Q","S"],i%3)))
        df.loc[::9, "Age"] = np.nan
        df.to_csv(path/"train.csv",index=False)
        df.drop(columns="Survived").iloc[:24].to_csv(path/"test.csv",index=False)
    if lesson in ("08", "09", "10"):
        path = work / "paddy-disease-classification"
        for label in ("normal", "bacterial_panicle_blight"):
            image_files(path/"train_images"/label, [f"{i}.jpg" for i in range(40)])
        names = [f"{i}.jpg" for i in range(5)]
        image_files(path/"test_images",names)
        pd.DataFrame({"image_id":names,"label":""}).to_csv(path/"sample_submission.csv",index=False)
        pd.DataFrame({"image_id":names,"label":"normal"}).to_csv(path/"train.csv",index=False)
    if lesson == "nlp":
        path = work/"us-patent-phrase-to-phrase-matching"
        path.mkdir()
        df = pd.DataFrame({"id":list(range(20)), "anchor":"steel beam", "target":["metal","iron","wood","gold"]*5,
                           "context":"A47", "score":np.arange(20)%5/4})
        df.to_csv(path/"train.csv",index=False)
        df.drop(columns="score").iloc[:4].to_csv(path/"test.csv",index=False)


def check_structure(path):
    ref = nbformat.read(ROOT / path.relative_to(HERE), as_version=4)
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert len(ref.cells) == len(nb.cells)
    for field in (ref.metadata.keys() | nb.metadata.keys()) - {"kernelspec", "language_info"}:
        assert ref.metadata.get(field) == nb.metadata.get(field), (path, field)
    transform = TransformerManager()
    for original, updated in zip(ref.cells, nb.cells, strict=True):
        for field in ("cell_type", "id", "metadata", "attachments"):
            assert original.get(field) == updated.get(field), (path, field)
        if updated.cell_type == "code":
            assert updated.execution_count is None and not updated.outputs
            compile(transform.transform_cell(updated.source), str(path), "exec")
    return nb


def test_source(lesson, i, s):
    """Select the smoke path in memory. i is the corresponding lecture cell index."""
    if lesson in ("00", "birds"):
        s=s.replace("use_local_photos = False","use_local_photos = True")
        if s.startswith(("learn = vision_learner(dls, resnet18,",)):
            return s.splitlines()[0]
    if lesson == "02":
        if "untar_data(" in s: return "path = Path('pets/images')"
        if "learn.fine_tune(" in s: return s.splitlines()[0]
    if lesson == "05" and " = train_model(" in s:
        s=re.sub(r"train_model\((18, )?", "train_model(1, ",s)
    if lesson == "06":
        if i in (19,21): return ""
        if i==38: return "learns = [preds, preds.clone()]"
    if lesson == "07":
        if i==84: return s.replace("range(100)","range(2)")
        if i==88: return s.replace("Classifier(100,","Classifier(2,")
    if lesson in ("08", "09"):
        if i in ((22,24) if lesson=="08" else (18,22,25,41)): return ""
        if "vision_learner(" in s:
            s=s.replace("metrics=error_rate","pretrained=False, metrics=error_rate")
            s=s.replace("    learn.fine_tune(epochs, 0.01)\n","")
    if lesson=="10":
        if i==10:
            return s.replace("metrics=error_rate", "pretrained=False, metrics=error_rate").replace(
                "learn.fit_one_cycle(epochs, 0.01)", "return learn")
        if i==12:
            return "probe = " + s + """
assert any(isinstance(cb, GradientAccumulation) and cb.n_acc==64 for cb in probe.cbs)
probe.model.eval()
with torch.no_grad():
    xb, yb = probe.dls.valid.one_batch()
    assert probe.model(xb[:1]).shape == (1, len(probe.dls.vocab))
del probe"""
        if i in (17,19,23,24,26,28,29): return ""
        if i==37: return """import timm
assert all(timm.get_pretrained_cfg(arch).url for arch in models)
assert GradientAccumulation(64).n_acc == 64
# Eight fixture prediction tensors for the eight original ensemble configurations.
tta_res = [(torch.arange(10).reshape(5, 2).float() + i, None) for i in range(8)]"""
    if lesson=="birds":
        s=s.replace("untar_data(URLs.CAMVID_TINY)", "Path('camvid')")
        for name in ("ADULT_SAMPLE", "ML_SAMPLE"):
            s=s.replace(f"untar_data(URLs.{name})", "Path('.')")
        s=s.replace("unet_learner(dls, resnet34)", "unet_learner(dls, resnet34, pretrained=False)")
        s="\n".join(line for line in s.splitlines() if not line.startswith(("learn.fine_tune(","learn.fit_one_cycle(")))
    if lesson=="nlp":
        if i==86: return s.replace("as_frame=True", "as_frame=True, download_if_missing=False")
        if i==118:
            s=s.replace("model = AutoModelForSequenceClassification.from_pretrained(model_nm, revision=model_revision, num_labels=1)",
                        """from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_nm, revision=model_revision, num_labels=1)
config.hidden_size, config.intermediate_size = 32, 48
config.pooler_hidden_size = config.hidden_size
config.num_hidden_layers, config.num_attention_heads = 1, 2
model = AutoModelForSequenceClassification.from_config(config)""")
        if i==120:return "metrics = trainer.evaluate()\nassert 'eval_pearson' in metrics"
    return s


CHECKS = {
    "00":"assert list(dls.vocab) == ['bird','forest'] and len(probs)==2",
    "01":"assert (a,b,c,d)==(1,2,4,8)",
    "02":"restored = load_learner('model.pkl')\nassert len(restored.predict(get_image_files(path)[0])[2])==2",
    "03":"assert len(df)>100 and len(tdf)>100 and len(show_subs(tdf,'Training','train_img_size').data)>1",
    "04":"assert abc.grad is not None and torch.isfinite(loss)",
    "05":"assert t_indep.shape==(120,12) and np.isfinite(float(acc(coeffs)))",
    "06":"assert preds.shape==(24,2) and Path('ens_sub.csv').is_file()",
    "07":"assert len(trees)==2 and len(rf.estimators_)==2 and '<svg' in draw_tree(m,trn_xs).pipe(format='svg').decode()",
    "08":"assert len(results)==5 and len(dls.vocab)==2",
    "09":"assert len(results)==5 and tta_preds.shape[1]==2",
    "10":"assert torch.equal(avg_pr, torch.stack(tta_prs).mean(0)) and len(tta_prs)==10 and len(ss)==5",
    "birds":"assert len(probs)==2 and '<svg' in gv2('a->b').pipe(format='svg').decode()",
    "nlp":"assert len(dds['train'])==15 and len(dds['test'])==5 and args.get_warmup_steps(100)==10\nassert np.isfinite(metrics['eval_pearson'])",
}


def run(path):
    nb = check_structure(path)
    lesson = path.name[:2] if path.parent.name!="slides" else path.stem.split("-")[0]
    if lesson in ("00", "02", "birds"):
        from torch.hub import get_dir
        if not (Path(get_dir())/"checkpoints"/"resnet18-f37072fd.pth").is_file():
            raise FileNotFoundError("Run the pretrained lesson 00 check once to cache ResNet-18.")
    lecture = nbformat.read(ROOT/path.name,as_version=4) if path.parent.name=="clean" else nb
    indices = iter(i for i,c in enumerate(lecture.cells) if c.cell_type=="code")
    for c in nb.cells:
        if c.cell_type=="code":
            c.source=test_source(lesson,next(indices),c.source)
    setup = f"import sys\nfrom pathlib import Path\nassert Path(sys.executable).resolve()==Path({sys.executable!r}).resolve()"
    # Keep image preparation cheap on Windows, where each worker imports torch.
    setup += "\nfrom fastcore.basics import defaults\ndefaults.cpus = 0"
    nb.cells.insert(0,nbformat.v4.new_code_cell(setup))
    nb.cells.append(nbformat.v4.new_code_cell(CHECKS[lesson]))
    if lesson == "nlp":
        original = nbformat.read(ROOT/"slides"/path.name, as_version=4)
        for i in (39, 41):
            demo = original.cells[i]
            tokens = ast.literal_eval(demo.outputs[0].data["text/plain"])
            nb.cells.append(nbformat.v4.new_code_cell(f"assert {demo.source} == {tokens!r}"))
    with tempfile.TemporaryDirectory() as tmp:
        fixtures(lesson,Path(tmp))
        client=NotebookClient(nb,kernel_name="course22",timeout=180)
        client.execute(cwd=tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Individual copies to test (default: all 23)")
    args = parser.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false"
    )
    paths = [path.resolve() for path in args.paths]
    if not paths:
        for source in sorted(ROOT.glob("*.ipynb")):
            if not source.name.startswith("00-"):
                paths.append(HERE / source.name)
            paths.append(HERE / "clean" / source.name)
        paths.extend(sorted((HERE / "slides").glob("*.ipynb")))

    failed = []
    for path in paths:
        name = path.relative_to(HERE)
        print("RUN", name, flush=True)
        try:
            run(path)
        except Exception as exc:
            print("FAILED", name, str(exc), flush=True)
            failed.append(path)
        else:
            print("PASS", name, flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} notebook(s) failed")


if __name__ == "__main__":
    main()

