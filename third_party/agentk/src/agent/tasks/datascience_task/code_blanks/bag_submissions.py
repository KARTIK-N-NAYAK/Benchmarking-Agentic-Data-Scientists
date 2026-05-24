from collections import defaultdict
import os
import pathlib
from argparse import ArgumentParser
from typing import Tuple, Optional, Any, Dict, Union

import numpy as np
import pandas as pd
from agent.tasks.datascience_task.utils import FileMap

try:
    from bag_code import bag
except ImportError:
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    print(f"Bag should be in {current_dir}")
    raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--submissions", nargs='+')
    parser.add_argument("--workspace", nargs=1)
    args = parser.parse_args()
    workspace = args.workspace[0]
    args = args.submissions
    args_lis = args[0].replace(' ', '').replace('[', '').split(',')
    sub_dict = defaultdict()
    for i in range(len(args_lis)):
        sub_dict[f'key{i+1}'] = pd.read_csv(f'{args_lis[i]}')
    final_sol = bag(sub_dict)
    final_sol.to_csv('./final_submission.csv')