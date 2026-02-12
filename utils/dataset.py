import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import random
from modelscope.msdatasets import MsDataset
import evaluate
from evaluate.utils.file_utils import DownloadConfig
from datasets import Dataset
import torch.utils.data


def get_glue_dataset(subset_name: str):

    mrpc = MsDataset.load("glue", subset_name=subset_name)

    ret: dict[str, Dataset] = {
        "train": mrpc["train"].ds_instance,
        "validation": mrpc["validation"].ds_instance,
        "test": mrpc["test"].ds_instance,
    }

    metric = evaluate.load(
        "glue",
        subset_name,
    )

    return ret, metric
