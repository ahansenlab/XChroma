# xbench

Array-based metrics for comparing predicted and measured contact maps.
The release contains the numerical metrics for **whole map**, **fine structure
(DoG)** and **focal loop strength at experimentally defined sites** evaluation.

See [metrics and examples](metrics/README.md) for input conventions and Python use.

```bash
python -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -r xbench/requirements.txt
python -m unittest discover -s xbench/tests -v
```

Run these commands and the Python examples from the repository root. Python
3.10–3.12 is supported. The metrics run on CPU and do not require model weights.
