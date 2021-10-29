# bert-benchmark

Work in progress

## Installation

```bash
# 1. Run TensorRT Release docker
$ docker run -v $HOME:/mnt --gpus all --rm -ti nvcr.io/nvidia/tensorrt:21.09-py3

# 2. Git clone
$ git clone https://github.com/siahuat0727/bert-benchmark && cd bert-benchmark

# 3. Install requirements
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py && \
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 4. Install nnfusion
$ cd /workspace && git clone https://github.com/microsoft/nnfusion.git /workspace/nnfusion --branch master --single-branch && DEBIAN_FRONTEND="noninteractive" bash /workspace/nnfusion/maint/script/install_dependency.sh && cd /workspace/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && cd /workspace

# Optional, use venv
$ python3 -m venv env --without-pip && . env/bin/activate && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py 
```

## Run benchmark

Including PyTorch, PyTorch-JIT, ONNXRuntime, TensorRT, DeepSpeed

```bash
$ bash run.sh
```

## Plot results

```bash
$ python3 plot.py  --files speed*.csv
Save inference-speed-python.png

$ python3 plot.py  --files nvprof*.csv  --frmt nvprof
Save inference-speed-nvprof.png
```
