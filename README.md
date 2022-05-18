# MIMO_TensorRT
Using TensorRT inference MIMO-UnetPlus on the jetson TX2, but I don't know to process output tensor.

## step1:

​	we need have MIMO-UnetPlus's model in onnx format. According to official guidance, we can get MIMO-UnetPlus's model in onnx format form model of the pytorch framework. `GetOnnx.py` has detail codes.

​	Execute the following statement in the code directory:

`python3 GetOnnx.py`

**Note**: You must have MIMO-UnetPlus's model of the pytorch framework in this directory.

## step2:

​	Using trtexec generate TensorRT engine.

​	`cd MIMO_Tensorrt/weights`

​	`trtexec --onnx='your_model'.onnx --verbose --explicitBatch --shapes=input_name:1x3x720x1080 --saveEngine='engineName'.engine`

**Note**: Make sure your environment variables are set.

## step3:

​	Begin to complie and run.

​	`cd ..`

​	`mkdir build`

​	`cd build`

​	`cmake ..`

​	`make`

​	`./MIMO_Tensorrt`

