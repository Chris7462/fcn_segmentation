# fcn_segmentation

## Generate the ONNX file
This will generate the onnx file in the `model` directory.
```python
python3 script/export_fcn_to_onnx.py --width 1238 --height 374 --model fcn_resnet50 --output-dir models
```
Note: The model can be `fcn_resnet50` or `fcn_resnet101`.

## Convert to TensorRT engine
```bash
trtexec --onnx=fcn_resnet50_1238x374.onnx --saveEngine=fcn_resnet50_1238x374.engine
```