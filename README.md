# bert-benchmark

Work in progress

## Installation

```bash
# 1. Run TensorRT Release docker
$ docker run -v $HOME:/mnt --gpus all --rm -ti nvcr.io/nvidia/tensorrt:21.10-py3

# 2. Git clone
$ git clone https://github.com/siahuat0727/bert-benchmark && \
git clone https://github.com/NVIDIA/TensorRT

# 3. Install requirements
$ cd bert-benchmark && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py && \
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
cd ..

# 4. Install nnfusion
$ git clone https://github.com/microsoft/nnfusion.git /workspace/nnfusion --branch master --single-branch && \
DEBIAN_FRONTEND="noninteractive" bash /workspace/nnfusion/maint/script/install_dependency.sh && \
cd /workspace/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && make install && cd /workspace

# Optional, use venv
$ python3 -m venv env --without-pip && . env/bin/activate && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py
```
## Verify correctness

```bash
$ bash test.sh && python3 plot.py --files max_abs_error_*.txt --type error
...
Save inference-max-abs-error.png
```

## Run benchmark

Including PyTorch, PyTorch-JIT, ONNXRuntime, TensorRT, DeepSpeed, NNFusion

```bash
$ bash run.sh && python3 plot.py --files speed*.csv && python3 plot.py --files nvprof*.csv  --frmt nvprof
...
Save inference-speed-python.png
...
Save inference-speed-nvprof.png
```

