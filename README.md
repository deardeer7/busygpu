# busygpu
keep your gpu busy

## Guidance
1. make sure you have pytorch and other essentials installed
2. just run it: `python run.py`

Note:
the code was tested on a NVIDIA A100 80GB GPU, and:
1. `run.py` achieves 90% gpu utilization
2. `run_10.py` achieves 10% and `run_40.py` 40%
3. you may adjust the parameters like：`model architecture`, `batch size`, `sleep time` to meet your needs.
