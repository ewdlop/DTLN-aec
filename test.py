import tflite2onnx

tflite_path = 'DTLN-aec/dtln_aec_512_1.tflite'
onnx_path = 'DTLN-aec/dtln_aec_512_1.onnx'

tflite2onnx.convert(tflite_path, onnx_path)