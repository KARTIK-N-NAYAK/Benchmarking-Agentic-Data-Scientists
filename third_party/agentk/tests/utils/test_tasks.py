import argparse
import csv
import os
import random
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from pyrootutils import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from agent.run_pipelines import run_setup_and_main_pipline
from agent import PROJECT_ROOT


class SyntheticTask(ABC):
    def __init__(self, output_dir: str):
        """
        Initialize the synthetic task base class.
        Args:
            output_dir: Directory where generated data and metadata will be saved.
        """
        self.output_dir = output_dir
        self.samples_per_class = 10
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def create_dataset(self, sub_dir_name: str, create_sample_submission: bool = False) -> None:
        """
        Create and save a synthetic dataset. Must be implemented by subclasses.

        Args:
            sub_dir_name: Name of the subdirectory where the dataset will be saved (e.g., 'train' or 'test').
            create_sample_submission: Whether to generate a sample submission file (default: False).

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_metadata(self) -> tuple[str, str, str]:
        """
        Creates task metadata.
        Must be implemented by subclasses to provide descriptions for the task, data, and metric.
        Returns:
            tuple: A tuple of strings (task_description, metric_description, data_description).
        """
        pass

    def write_metadata(self) -> None:
        """
        Write metadata files to the output directory.
        Creates the following files:
            - raw_task_description.txt
            - raw_metric_description.txt
            - raw_data_description.txt
        Each file contains the corresponding text returned from `get_metadata()`.
        Returns:
            None
        """
        task_desc, metric_desc, data_desc = self.get_metadata()
        files = {
            "raw_task_description.txt": task_desc,
            "raw_metric_description.txt": metric_desc,
            "raw_data_description.txt": data_desc,
        }
        for filename, content in files.items():
            with open(os.path.join(self.output_dir, filename), "w") as f:
                f.write(content)

    @final
    def run(self) -> None:
        """
        Execute the full synthetic task pipeline.
        This method:
            - Creates training and testing datasets
            - Generates a sample submission file for the test set
            - Writes metadata description files

        This method should not be overridden in subclasses.
        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.create_dataset(sub_dir_name="train", create_sample_submission=False)
        self.create_dataset(sub_dir_name="test", create_sample_submission=True)
        self.write_metadata()


class CVSyntheticTask(SyntheticTask):
    """
    A synthetic computer vision (CV) classification task.
    This task generates two types of synthetic images:
    - red_circle: A red circle on a white background
    - blue_square: A blue square on a white background

    It creates train/test image folders along with corresponding CSV files
    and metadata descriptions to simulate a simple image classification workflow.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the CVSyntheticTask task.
        Args:
            output_dir:  Path to the directory where generated data will be saved.
        """
        super().__init__(output_dir=output_dir)
        self.class_names = ['red_circle', 'blue_square']
        self.image_shape = (64, 64)

    def create_dataset(self, sub_dir_name: str, create_sample_submission: bool = False) -> None:
        """
        Create a synthetic image dataset and save its mapping to a CSV file.
        Args:
            sub_dir_name: Subdirectory under output_dir to store dataset (e.g., 'train' or 'test').
            create_sample_submission: If True, generate a sample_submission.csv without labels.

        Returns:
            None
        """
        data_map = os.path.join(self.output_dir, f'{sub_dir_name}.csv')
        with open(data_map, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "image_path", "label"])

            for c in self.class_names:
                os.makedirs(os.path.join(self.output_dir, sub_dir_name, c), exist_ok=True)
                for i in range(self.samples_per_class):
                    img = Image.new("RGB", self.image_shape, color=(255, 255, 255))
                    draw = ImageDraw.Draw(img)

                    x0, y0 = random.randint(10, 20), random.randint(10, 20)
                    x1, y1 = x0 + 30, y0 + 30

                    if c == self.class_names[0]:
                        draw.ellipse([x0, y0, x1, y1], fill=(255, 0, 0), outline=None)
                    elif c == self.class_names[1]:
                        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 255), outline=None)

                    filename = f"{c}_{i}.png"
                    image_path = os.path.join(self.output_dir, sub_dir_name, c, filename)
                    img.save(image_path)

                    writer.writerow([filename, image_path, c])

        if create_sample_submission:
            submission_df = pd.read_csv(data_map)
            submission_df.drop(columns=['image_path', 'label'], inplace=True)
            submission_df['label'] = None
            create_sample_submission_name = os.path.join(self.output_dir, 'sample_submission.csv')
            submission_df.to_csv(create_sample_submission_name, index=False)

    def get_metadata(self) -> tuple[str, str, str]:
        """
        Return the task description, metric, and data descriptions for the image classification task.
        Returns:
            tuple[str, str, str]: A tuple containing:
                - task_description: Description of the classification task
                - metric_description: Description of the evaluation metric(s)
                - data_description: Explanation of the dataset structure and files
        """
        task_description = """
        Red Circle vs Blue Square Image Classification

        Description:
        This is an image classification task. The dataset contains two visual classes:

        red_circle: Images with a red circle on a white background.

        blue_square: Images with a blue square on a white background.

        Each image is 64×64 pixels.
        The goal is to train a classifier that can distinguish between the two based on shape and color.
        """

        data_description = """
        You are provided with test and training images and train.csv and test.csv for the corresponding images. 
        There are two classes red_circle and blue_square. 
        """

        metric_description = f"Accuracy can be used a  metric for this task."

        return task_description, metric_description, data_description


class NLPSyntheticTask(SyntheticTask):
    """
    A synthetic NLP sentiment classification task.
    This task generates short, labeled text samples representing synthetic product reviews.
    The labels are either 'positive' or 'negative', and the goal is to create a classification dataset.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the NLPSyntheticTask.
        Args:
            output_dir:  Path to the directory where the generated data and metadata will be stored.
        """
        super().__init__(output_dir=output_dir)

    def create_dataset(self, sub_dir_name: str, create_sample_submission: bool = False) -> None:
        """
        Create a synthetic sentiment classification dataset and save it to a CSV file.
        Args:
            sub_dir_name: Subdirectory under output_dir to store dataset (e.g., 'train' or 'test').
            create_sample_submission: If True, generate a sample_submission.csv without labels.

        Returns:
            None
        """
        positive_seeds = [
            "I love this!",
            "This is amazing",
            "Absolutely wonderful",
            "I'm very satisfied",
            "Best product ever",
        ]

        negative_seeds = [
            "I hate this",
            "This is terrible",
            "Really disappointing",
            "Not worth it",
            "Worst experience ever",
        ]

        data = []

        for _ in range(self.samples_per_class):
            data.append((random.choice(positive_seeds), "positive"))
            data.append((random.choice(negative_seeds), "negative"))

        # Save to CSV
        df = pd.DataFrame(data, columns=["text", "label"])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        data_map = os.path.join(self.output_dir, f'{sub_dir_name}.csv')
        df.to_csv(data_map, index=False)
        if create_sample_submission:
            submission_df = pd.read_csv(data_map)
            submission_df = submission_df[['id']].copy()
            submission_df['label'] = None

            create_sample_submission_name = os.path.join(self.output_dir, 'sample_submission.csv')
            submission_df.to_csv(create_sample_submission_name, index=False)

    def get_metadata(self) -> tuple[str, str, str]:
        """
        Return the task description, metric, and data descriptions for the classification task.
        Returns:
            tuple[str, str, str]: A tuple containing:
                - task_description: Description of the classification task
                - metric_description: Description of the evaluation metric(s)
                - data_description: Explanation of the dataset structure and files
        """
        task_description = """
        Sentiment Classification on Synthetic Reviews

        Description:
        This is a text classification task. The dataset contains synthetic product reviews labeled as either:

        positive: Reviews expressing satisfaction or praise.

        negative: Reviews expressing dissatisfaction or complaints.

        Each example consists of a short sentence (e.g., "I loved the product!" or "It was a waste of money").
        The goal is to train a classifier that can correctly predict the sentiment of a review.
        """

        data_description = """
        You are provided with train.csv and test.csv containing short synthetic product reviews.
        Each row contains a text sample and its corresponding sentiment label (positive or negative).
        """

        metric_description = "Accuracy can be used as a metric for this task."

        return task_description, metric_description, data_description


class TabularSyntheticTask(SyntheticTask):
    """
    A synthetic tabular classification task for income prediction.
    This task generates a binary classification dataset with categorical and numerical features to predict
    whether an individual's income is high or low based on attributes such as age, education, occupation, and working hours.
    """

    def __init__(self, output_dir: str, num_samples=20) -> None:
        """
        Initialize the TabularSyntheticTask.
        Args:
            output_dir: Path to the directory where the generated data and metadata will be stored.
            num_samples: Total number of synthetic samples to generate.
        """
        super().__init__(output_dir=output_dir)
        self.num_samples = num_samples

    def create_dataset(self, sub_dir_name: str, create_sample_submission: bool = False) -> None:
        """
        Generate synthetic tabular data and save it as a CSV file.
        Args:
            sub_dir_name: Subdirectory under output_dir to store dataset (e.g., 'train' or 'test').
            create_sample_submission: If True, generate a sample_submission.csv without labels.

        Returns:
            None
        """
        np.random.seed(42)
        education_levels = ['high_school', 'bachelor', 'master', 'phd']
        occupation_types = ['tech', 'sales', 'admin', 'blue_collar', 'manager']

        df = pd.DataFrame(
            {
                "age": np.random.randint(18, 65, self.num_samples),
                "education_level": np.random.choice(education_levels, self.num_samples),
                "occupation_type": np.random.choice(occupation_types, self.num_samples),
                "hours_per_week": np.random.randint(20, 60, self.num_samples),
                "label": np.random.choice(["high_income", "low_income"], self.num_samples, p=[0.4, 0.6])
            }
        )
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        data_map = os.path.join(self.output_dir, f'{sub_dir_name}.csv')
        df.to_csv(data_map, index=False)
        if create_sample_submission:
            submission_df = pd.read_csv(data_map)
            submission_df = submission_df[['id']].copy()
            submission_df['label'] = None

            create_sample_submission_name = os.path.join(self.output_dir, 'sample_submission.csv')
            submission_df.to_csv(create_sample_submission_name, index=False)

    def get_metadata(self) -> tuple[str, str, str]:
        """
        Return metadata for the synthetic tabular income prediction task.
        Returns:
            tuple[str, str, str]: A tuple containing:
                - task_description: Description of the classification task
                - metric_description: Description of the evaluation metric(s)
                - data_description: Explanation of the dataset structure and files
        """
        task_description = """
        Synthetic Tabular Data Classification: Income Prediction

        Description:
        This is a binary classification task on tabular data. Each record represents a individual with attributes such as:

        - age
        - education_level
        - occupation_type
        - hours_per_week

        The label indicates whether the person's income is:

        high_income: Annual income greater than $50K  
        low_income: Annual income $50K or below

        The goal is to train a model that predicts income class based on individual features.
        """

        data_description = """
        You are provided with synthetic tabular data in train.csv and test.csv format.
        Each row includes numeric and categorical features and a label indicating income class (high_income or low_income).
        """

        metric_description = "accuracy can be used as metrics for this task."

        return task_description, metric_description, data_description


def set_cuda_device() -> None:
    """
    Ensures that only one GPU is visible to the script by setting the CUDA_VISIBLE_DEVICES environment variable.
    Returns:
        None
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Running the test on the CPU.")
        return

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("CUDA_VISIBLE_DEVICES was not set, defaulting to GPU 0")
    else:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            devices_list = visible_devices.split(",")
            if len(devices_list) == 1:
                print(f"CUDA_VISIBLE_DEVICES is set to GPU {devices_list[0]}")
            else:
                first_device = devices_list[0].strip()
                os.environ["CUDA_VISIBLE_DEVICES"] = first_device
                print(f"CUDA_VISIBLE_DEVICES was set, narrowed down to GPU {first_device}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("CUDA_VISIBLE_DEVICES was empty, defaulting to GPU 0")


def run_pipeline_for_task(task_id, task_class, task_type, raw_data_dir, run_setup_only) -> dict[str, ...]:
    """
    Run setup or main pipeline for a given task.
    Args:
        task_id: Task name
        task_class: The class of the synthetic task (CVSyntheticTask, NLPSyntheticTask, etc.)
        task_type: Modality of the task ('cv', 'nlp', or 'tabular')
        raw_data_dir: Base directory
        run_setup_only: True for setup only run
    Returns:
        Status of the pipline test.
    """
    task_output_dir = f"{raw_data_dir}/{task_type}/raw_data/{task_id}"

    task = task_class(output_dir=task_output_dir)
    task.run()

    prep_task = "data_preprocessing"
    prep_method = "data-prep-flow"
    llm = "hf_hub/gpt2-small"
    ds_method = "agent-k-solve"

    working_dir = Path(raw_data_dir) / task_type
    response_file_map = dict(cv='img', nlp='txt', tabular='tab')
    response_file_tag = response_file_map[task_type]
    scripted_answer_dir = PROJECT_ROOT / "third_party" / "data_science" / "benchmark_test"
    pre_reg_response_setup = Path(__file__).parent / f"test_{task_type}_task.txt"
    pre_reg_response_main = scripted_answer_dir / f"input_{response_file_tag}_target_tab.txt"

    status = run_setup_and_main_pipline(
        workspace_name=str(working_dir),
        task_id=task_id,
        prep_task=prep_task,
        prep_method=prep_method,
        ds_method=ds_method,
        llm=llm,
        code_llm=llm,
        is_local_task=True,
        is_tabular=(task_type == "tabular"),
        total_time=600,  # 10 Minutes
        max_setups=1,
        alt_raw_data_root=f"{raw_data_dir}/{task_type}/raw_data",
        setup_default_response_path=str(pre_reg_response_setup),
        main_pipeline_default_response_path=str(pre_reg_response_main),
        max_cpu=0,
        allow_default_response=True,
        debug_mode=True,
        terminate_after_training=True,
        run_setup_only=run_setup_only,
        use_final_unit_test=True,
        attempt=None,
        attempt_spec="",
        use_ci_handling=False,
        blend_after_n=3,
        max_time_per_submission=1800

    )

    return status


def main(is_setup_pipeline_test: bool, task_class_name: str | None = None) -> int:
    """
    Generates synthetic datasets for Tabular, Computer Vision (CV), and Natural Language Processing (NLP) tasks,
    Creates setups for the three tasks along with the final unit test pipeline.

    Workflow:
        - Sets environment variable to limit usage to a single GPU (GPU 0).
        - Creates a temporary directory to store raw data and outputs for each task.
        - Instantiates synthetic task objects for CV, NLP, and Tabular modalities.
        - For each task:
            - Runs the dataset creation and metadata writing process.
            - Executes the setup pipeline (and the main pipeline) and verifies if the pipeline(s) completed successfully.
        - Prints summary indicating the success or failure of each task's setup pipeline test.
    Args:
        is_setup_pipeline_test: Whether to run only the setup pipeline test.
        task_class_name: Specify the task to be run , by default runs all the tasks.

    Returns:
        Status of the complete pipline test (CV, NLP and Tabular). 0 for success, 1 for failure.
    """
    # Set proper GPU for test
    set_cuda_device()

    test_output = 0
    task_class_map = {
        "CVSyntheticTask": (CVSyntheticTask, "cv"),
        "NLPSyntheticTask": (NLPSyntheticTask, "nlp"),
        "TabularSyntheticTask": (TabularSyntheticTask, "tabular"),
    }

    # Determine which tasks to run
    if task_class_name:
        if task_class_name not in task_class_map:
            raise ValueError(
                f"Unknown task class: {task_class_name}. Available: {list(task_class_map.keys())}"
            )
        tasks = [task_class_map[task_class_name]]
    else:
        tasks = list(task_class_map.values())

    with tempfile.TemporaryDirectory() as tmp_dirname:
        raw_data_dir = tmp_dirname
        print(raw_data_dir)
        task_id = "custom_task_test"

        test_status = {}
        for task_class, task_type in tasks:
            print("Running task:", task_type)
            pipeline_status = run_pipeline_for_task(
                task_id=task_id,
                task_class=task_class,
                task_type=task_type,
                raw_data_dir=raw_data_dir,
                run_setup_only=is_setup_pipeline_test
            )
            test_status[task_type] = pipeline_status

        # Check results
        for task_type, run_status in test_status.items():
            if is_setup_pipeline_test:
                if run_status['setup']:
                    print(f"✅ Setup pipeline test successful for {task_type} task.")
                else:
                    print(f"❌ Setup pipeline test failed for {task_type} task.")
                    test_output = 1
            else:
                if run_status['setup']:
                    print(f"✅ Setup pipeline test successful for {task_type} task.")
                    if 'main' in run_status and run_status['main']:
                        print(f"✅ Main pipeline test successful for {task_type} task.")
                    else:
                        print(f"❌ Main pipeline test failed for {task_type} task.")
                        test_output = 1
                else:
                    print(f"❌ Setup pipeline test failed for {task_type} task.")
                    test_output = 1

    return test_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run setup and main pipeline.")
    parser.add_argument(
        "--setup_pipeline", action='store_true', default=False,
        help='Test setup pipeline only.'
    )
    parser.add_argument(
        "--task_class", type=str,
        help="Name of the task class to run (CVSyntheticTask, NLPSyntheticTask, TabularSyntheticTask)."
    )
    args_ = parser.parse_args()

    exit_code = main(is_setup_pipeline_test=args_.setup_pipeline, task_class_name=args_.task_class)
    exit(exit_code)
