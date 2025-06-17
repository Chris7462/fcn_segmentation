# fcn_segmentation

## Generate the ONNX file
This will generate the onnx file in the `model` directory.
```python
python3 script/export_fcn_to_onnx.py --width 1238 --height 374 --model fcn_resnet50 --output-dir model
```
Note: The model can be `fcn_resnet50` or `fcn_resnet101`.
