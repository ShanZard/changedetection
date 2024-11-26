import evaluate

import os
# 设置Hugging Face的国内镜像源
os.environ["HUGGINGFACE_HUB_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-hub"
os.environ["HF_DATASETS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-datasets"
os.environ["HF_METRICS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-metrics"
os.environ["HF_HOME"] = "/path/to/your/hf_home_directory"
os.environ["TRANSFORMERS_CACHE"] = "/path/to/your/transformers_cache_directory"
os.environ["HF_DATASETS_CACHE"] = "/path/to/your/datasets_cache_directory"
accuracy = evaluate.load("/private/evaluate/metrics/accuracy/accuracy.py")
f1 = evaluate.load("f1")
clf_metrics = combine(["accuracy", "f1"])