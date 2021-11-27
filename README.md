# ReCLIP: A Strong Zero-shot Baseline for Referring Expression Comprehension

## Setup
First, install `pytorch`, `torchvision`, and `cudatoolkit` following the instructions in `https://pytorch.org/get-started/locally/`. Then run `pip install -r requirements.txt`. Finally, run `pip install git+https://github.com/openai/CLIP.git`. Download the [ALBEF pre-trained checkpoint](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) and place it at the path `albef/checkpoint.pth`.
Download the images for RefCOCO/g/+ from [http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/train2014.zip). Download the images for RefGTA from this [Google Drive folder](https://drive.google.com/drive/folders/1pcdwA--xSAkbsOwjqhhyXMRZH7_sjQXU).

## Experiments
The following format can be used to run experiments:
```
python main.py --input_file INPUT_FILE --image_root IMAGE_ROOT --method {baseline/gradcam/parse/random} --baseline_head --gradcam_alpha 0.5 --possessive --no_ternary {--clip_model RN50x16,ViT-B/32} {--albef_path albef --albef_mode itm/itc --albef_block_num 8/11} {--mdetr mdetr_efficientnetb3/mdetr_efficientnetb3_refcocoplus/mdetr_effcientnetb3_refcocog --freeform_bboxes} --box_representation_method crop,blur --box_area_threshold 0 --box_method_aggregator sum --square_size {--detector_file PATH_TO_DETECTOR_FILE}
```
(`/` is used above to denote different options for a given argument.) For ALBEF, we use ALBEF block num 8 for ITM (following the ALBEF paper) and block num 11 for ITC. Note that several arguments are only required for a particular "method," but they can still be included in the command when using a different method.

## Acknowledgements
The code in the `albef` directory is taken from the [ALBEF repository](https://github.com/salesforce/ALBEF/tree/main). The code in `clip_mm_explain` is taken from [https://github.com/hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability).