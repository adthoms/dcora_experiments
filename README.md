# dcora_experiments
Distributed Certifiably Correct Range-aided SLAM Experiments

## Getting Started

Access the evaluation pipeline interface via:

```bash
cd ~/dcora_experiments/scripts
python3 evaluation_pipeline.py --help
```

### Code Standards

Any necessary coding standards are enforced through `pre-commit`. This will run a series of `hooks` when attempting to commit code to this repo.

Set up `pre-commit` via:

```bash
cd ~/dcora_experiments
pip3 install pre-commit
pre-commit install
```