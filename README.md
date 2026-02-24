# thesis-explainable-face-verification
$env:PYTHONPATH="."
python scripts/torch_baseline_pair.py --imgA "assets\demo_images\img1.jpg" --imgB "assets\demo_images\img2.jpg" --outdir results\adaface_run --do_parsing
“You must clone AdaFace into third_party/adaface”

“Download checkpoint into pretrained/ (ignored by git)”