# bert-benchmark

Work in progress 

## Installation
 
```bash
# Run TensorRT Release docker
$ docker run -v $HOME:/mnt --gpus all --rm -ti nvcr.io/nvidia/tensorrt:21.09-py3

# Git clone
$ git clone https://github.com/siahuat0727/bert-benchmark && cd bert-benchmark

# Install requirements
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py && pip install -r requirements.txt

# OR
# Install requirements in venv
$ python3 -m venv env --without-pip && . env/bin/activate && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && rm get-pip.py && pip install -r requirements.txt
```

## Run benchmark

```bash
$ bash run.sh
$ python plot.py --files speed-*.csv
```
