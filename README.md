Breast cancer image classifier

This repository contains `train_and_predict.py`, a simple transfer-learning pipeline using MobileNetV2 to classify breast tissue images as malignant or normal.

Quick start (recommended in a virtual environment):

1. Install dependencies:
   pip install -r requirements.txt

2. Run a quick smoke training (fast):
   python train_and_predict.py --quick --epochs 2 --batch_size 16

3. For full training:
   python train_and_predict.py --epochs 10 --batch_size 16

4. To run prediction only with existing weights:
   python train_and_predict.py --mode predict_only --weights best_model.h5

Output:

- `best_model.h5` : best weights saved by ModelCheckpoint
- `submission.csv`: predictions for images in the `test/` folder, columns `image file,label` where label is `M` for malignant and `N` for normal.
- `best_model.keras` : best weights saved by ModelCheckpoint
- `submission.csv`: predictions for images in the `test/` folder, columns `image file,label` where label is `M` for malignant and `N` for normal.

Notes & tips

- Use a GPU and TensorFlow with GPU support for reasonable training speed.
- Increase `img_size` for improved accuracy, but it increases memory and training time.
- Consider cross-validation, test-time augmentation, ensembling, and stronger regularization to reduce overfitting on this small dataset.
