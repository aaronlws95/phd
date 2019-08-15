# Machine Learning

## Setup
```bash
./setup/init.sh /path/to/root/
```

### Root directory
```bash
Root
├── mlcv-exp
│   ├── data
│   │   ├── cfg
│   │   │   ├── model1
│   │   │   └── model2
│   │   ├── exp
│   │   │   ├── model1
│   │   │   └── model2
│   │   ├── labels
│   │   ├── weights
│   │   └── root.txt
│   ├── setup
│   ├── test
│   ├── main.py
│   └── src
└── datasets
    ├── dataset1
    └── dataset2
```

## Usage

* Training
```bash
python main.py --cfg <cfgfile> --mode train --epoch <optional>
```

* Inference
```bash
python main.py --cfg <cfgfile> --mode test --epoch <load_epoch> --split <optional>
```

* Evaluation

Use [evaluation.ipynb](notebooks/evaluation.ipynb) to evaluate models after inference