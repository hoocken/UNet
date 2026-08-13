# UNet Implementation for Torso Segmentation
The U-Net architecture is inspired from this [repo](https://github.com/milesial/Pytorch-UNet) and some adjustments are from the [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) paper. Namely, this uses instance normalization as the batch size is very small. The loss used is also taken from the nnU-Net paper, however this used generalized dice loss instead of the normal one.

## Usage
To run the segmentation, simply do
```
python run_segmentation.py run.input_image=<INPUT_FILE>
```

or set your own filepaths in `config/config.yaml`.