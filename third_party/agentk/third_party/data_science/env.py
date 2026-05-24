from __future__ import annotations

import abc
import dataclasses
import glob
import os
import shutil
import time
from datetime import datetime
from functools import partialmethod
from pathlib import Path

import pandas as pd
from agent import PROJECT_ROOT
from agent.tasks.datascience_task.utils import FileMap, replace_in_file
from agent.utils.utils import ListableEnum
from dataclasses_json import dataclass_json
from ds_agent.utils import save_w_pickle, load_w_pickle

from third_party.data_science.env_stages import TopDatascienceStage, DataScienceStageNames
from third_party.data_science.utils import get_path_to_ds_python

NEW_SUBMISSION_DIRNAME = "new_submission"
NEW_BLEND_DIRNAME = "new_blend_submission"
DS_SUBMISSIONS_DIRNAME = "submissions"
ID_COLUMN_NAME = "id"


class DSObsKey(ListableEnum):
    AVAILABLE_ACTIONS = "available_actions"
    TASK_DESCRIPTION = "task_description"
    DATA_DESCRIPTION = "data_description"
    SUMMARIZED_METRIC_DESCRIPTION = "metric_description"
    TABLE_TARGET_DESCRIPTION = "table_target_description"
    IMAGE_TARGET_DESCRIPTION = "image_target_description"
    TEXT_TARGET_DESCRIPTION = "text_target_description"
    SUBMISSION_LIST = "submission_list"
    CURRENT_SUBMISSION = "current_submission"
    CURRENT_SUBMISSION_PERF = "current_submission_perf"
    SENT_SUBMISSION_NAMES = "sent_submission_names"  # list of submissions already sent in the past for this competition
    SUBMISSION_SENT_SUCCESSFULLY = "submission_sent_successfully"


class DSActionHypKeys(str, ListableEnum):
    SUMMARY_STEP = "summary_step"
    SUBMISSION_SUMMARY = "submission_summary"
    CODE_BLANK = "code_blank"
    SUBMISSION_NAME = "submission_name"
    SUBMISSIONS_LIST = "submissions_list"


@dataclasses.dataclass
class DSAction:
    stage_name: DataScienceStageNames
    hyps: dict[DSActionHypKeys, ...]


@dataclass_json
@dataclasses.dataclass
class Status(abc.ABC):
    is_necessary: bool  # if this element is necessary
    is_available: bool  # if this element can be present
    is_done: bool  # if this element is done
    is_pending: bool  # if this element is started but not over yet
    specific_description: str = ""

    @staticmethod
    @abc.abstractmethod
    def generic_name() -> str:
        pass

    def to_str(self) -> str:
        if not self.is_available:
            return ""

        status = ""
        if self.is_done:
            status += "[DONE] "
        else:
            if self.is_necessary and not self.is_pending:
                status += "[TODO] "
            elif self.is_pending:
                status += "[PENDING] "
            else:
                status += "[TODO (OPTIONAL)] "

        status += f"{self.generic_name()}"
        if self.is_done:
            status += f": {self.specific_description}"
        return status


class ClassImbalanceStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Handle class imbalances"


class TableFEStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Feature engineering of raw tabular fields"


class TableModelStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Embedder for processed tabular fields"


class TableEmbedStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Embedder for raw tabular fields"


class ImgEmbedStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Embedder for images"


class TrainImgTransformStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Transform functions for train images"


class TestImgTransformStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Transform functions for test images"


class ImgModelStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Embedder for transformed images"


class TxtEmbedStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Embedder for text fields"


class TableHeadStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Logits and regression targets decoder"


class RegeressionTargetTransformStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Transform regression targets"


class ImgHeadStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Image decoder"


class TxtHeadStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Text decoder"


class BagStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Bagging submissions"


class BlendStatus(Status):
    @staticmethod
    def generic_name() -> str:
        return "Blending submissions"


@dataclass_json
@dataclasses.dataclass
class SubmissionState(abc.ABC):

    def __post_init__(self) -> None:
        self.summary: str | None = None

    @abc.abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError()


@dataclass_json
@dataclasses.dataclass
class ClassicalTabSubmissionState(SubmissionState):
    table_fe: TableFEStatus

    def to_str(self) -> str:
        status = ""
        if self.table_fe.is_available:
            status += f"- {self.table_fe.to_str()}\n"
        return status


@dataclass_json
@dataclasses.dataclass
class DNNSubmissionState(SubmissionState):
    class_imbalance: ClassImbalanceStatus
    table_fe: TableFEStatus
    table_model: TableModelStatus
    table_embed: TableEmbedStatus
    table_head: TableHeadStatus
    regression_target_transform: RegeressionTargetTransformStatus
    img_embed: ImgEmbedStatus
    train_img_transform: TrainImgTransformStatus
    test_img_transform: TestImgTransformStatus
    img_model: ImgModelStatus
    img_head: ImgHeadStatus
    txt_embed: TxtEmbedStatus
    txt_head: TxtHeadStatus
    bag: BagStatus
    blend: BlendStatus

    def to_str(self) -> str:
        status = ""
        if self.table_embed.is_available:
            status += f"- {self.table_embed.to_str()}\n"
            status += f"    - {self.table_fe.to_str()}\n"
            status += f"    - {self.table_model.to_str()}\n"
        if self.img_embed.is_available:
            status += f"- {self.img_embed.to_str()}\n"
            status += f"    - {self.train_img_transform.to_str()}\n"
            status += f"    - {self.test_img_transform.to_str()}\n"
            status += f"    - {self.img_model.to_str()}\n"
        if self.txt_embed.is_available:
            status += f"- {self.txt_embed.to_str()}\n"
        if self.class_imbalance.is_available:
            status += f"- {self.class_imbalance.to_str()}\n"
        if self.regression_target_transform.is_available:
            status += f"- {self.regression_target_transform.to_str()}\n"
        if self.table_head.is_available:
            status += f"- {self.table_head.to_str()}\n"
        if self.img_head.is_available:
            status += f"- {self.img_head.to_str()}\n"
        if self.txt_head.is_available:
            status += f"- {self.txt_head.to_str()}\n"
        return status


@dataclass_json
@dataclasses.dataclass(kw_only=True)
class DataScienceSubmissionCard:
    root_path: str
    name: str
    submission_state: SubmissionState | None

    def create(self) -> None:
        # create submission structure
        os.makedirs(self.path, exist_ok=True)
        self.save()

    def save(self) -> None:
        if self.submission_state is not None:
            submission_state_json = self.submission_state.to_json()
            with open(self.submission_state_path, "w") as f:
                f.write(submission_state_json)
            if self.submission_state.summary is not None:  # why is the summary not in the submission state json?
                with open(self.summary_path, "w") as f:
                    f.write(self.submission_state.summary)

    @staticmethod
    def get_summary_path(path: str) -> str:
        return os.path.join(path, FileMap.SUBMISSION_SUMMARY.value)

    @staticmethod
    def get_path(root_path: str, name: str) -> str:
        return os.path.join(root_path, name)

    @property
    def path(self) -> str:
        return self.get_path(root_path=self.root_path, name=self.name)

    @property
    def summary_path(self) -> str:
        return self.get_summary_path(path=self.path)

    @property
    def submission_state_path(self) -> str:
        return self.get_submission_state_path(path=self.path)

    @staticmethod
    def get_submission_state_path(path: str) -> str:
        """ Returns the path to the json containing submission info
        Args:
            path: path to the specific submission folder
        """
        return os.path.join(path, FileMap.SUBMISSION_STATE.value)

    @staticmethod
    def load(path: str) -> DataScienceSubmissionCard:
        submission_state = None
        submission_state_path = DataScienceSubmissionCard.get_submission_state_path(path=path)
        if os.path.exists(submission_state_path):
            with open(submission_state_path, "r") as f:
                submission_state_json = "".join(f.readlines())
            for state_class in [DNNSubmissionState, ClassicalTabSubmissionState]:
                try:
                    submission_state = state_class.from_json(submission_state_json)
                    summary_path = Path(DataScienceSubmissionCard.get_summary_path(path=path))
                    if summary_path.is_file():
                        with open(summary_path, "r") as f:
                            submission_state.summary = f.read()
                    break
                except:
                    pass
            if submission_state is None:
                raise RuntimeError(f"Could not load {submission_state_path}: {submission_state_json}")
        return DataScienceSubmissionCard(
            root_path=os.path.dirname(path),
            name=os.path.basename(path),
            submission_state=submission_state
        )

    def update_given_name(self, new_name: str) -> None:
        """ Read new name file and rename the submission """
        oldpath = self.path
        self.name = new_name
        os.rename(src=oldpath, dst=self.path)


def check_accuracy_classical_tab_fe(code_block: str, workspace_path: str) -> tuple[dict, int]:
    """ Find model accuracy based on the generated feature engineering code """
    import json
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, classification_report

    x_train = pd.read_csv(os.path.join(workspace_path, 'data', 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(workspace_path, 'data', 'y_train.csv'))

    x_test = pd.read_csv(os.path.join(workspace_path, 'data', 'X_val.csv'))
    y_test = pd.read_csv(os.path.join(workspace_path, 'data', 'y_val.csv'))

    interested_column = y_train.columns[-1]
    classification_regression_tag = interested_column.split('_')[-1]
    y_train = y_train[interested_column]
    y_test = y_test[interested_column]

    def write_output(output_dict_: dict) -> tuple[dict, int]:
        """ Write to json file """
        output_file_path = os.path.join(workspace_path, 'data', 'model_output.json')

        # Check if the file exists
        if os.path.exists(output_file_path):
            # Load the existing content
            with open(output_file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        else:
            # If the file does not exist, start with an empty dictionary
            existing_data = {}

        # Find the next available key (1-based index)
        next_key = max(map(int, existing_data.keys()), default=0) + 1

        # Append the new output_dict with the next key
        existing_data[next_key] = output_dict_

        # Write the updated dictionary back to the JSON file
        with open(output_file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        return existing_data, next_key

    output_dict = {}
    if classification_regression_tag == "regression":
        # do regression
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        output_dict = {
            'Mean Squared Error': mse,
            'R^2 Score': r2
        }

    elif classification_regression_tag == "classification":
        # do classification
        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        output_dict = {
            'Classification model accuracy': accuracy,
            'Classification Report': report
        }
    print(output_dict)
    output_dict.update({"code_block ": code_block})
    complete_output, num_iterations = write_output(output_dict_=output_dict)
    return complete_output, num_iterations


class ResumeCheckpoint:
    def __init__(self, max_exec_time: float, top_stage: TopDatascienceStage,
                 obs: dict[DSObsKey, ...], step_num: int) -> None:
        self.max_exec_time = max_exec_time
        self.top_stage = top_stage
        self.obs = obs
        self.step_num = step_num


class DataScienceEnv:

    def __init__(
            self, task_id: str, prepared_setup_dir: str, prepared_version: str, exp_id: str, workspace_path: str,
            terminate_after_training: bool = False, max_exec_time: int | None = None, is_local_task: bool = False
    ) -> None:
        """
        Args:
            task_id: ID of the task (e.g. name of the Kaggle competition)
            exp_id: ID for this specific run (e.g. contains timestamp)
            prepared_setup_dir: Absolute path to the prepared setup directory.
            prepared_version: ID of the data preparation. Subfolder name of prepared_setup_dir/task_id/
            workspace_path: path to where new code and temporary files should be created to fulfill the task 
        """
        self.task_id = task_id
        self.exp_id = exp_id
        self.prepared_setup_dir = os.path.abspath(prepared_setup_dir)
        self.prepared_version = prepared_version
        self.workspace_path = os.path.abspath(workspace_path)
        self.terminate_after_training = terminate_after_training
        self.max_exec_time = max_exec_time
        self.done = None
        self.obs: dict[DSObsKey, ...] = {}

        self.top_stage: TopDatascienceStage | None = None
        self.current_submission: DataScienceSubmissionCard | None = None
        self.start_time = None
        self.step_num = None
        self.is_local_task = is_local_task

    @property
    def submissions_path(self) -> str:
        return os.path.join(self.workspace_path, DS_SUBMISSIONS_DIRNAME)

    def get_submissions(self) -> dict[str, DataScienceSubmissionCard]:
        submissions = {}
        for subfolder in glob.iglob(self.submissions_path + "/*"):
            submission_name = os.path.basename(subfolder)
            if os.path.isfile(subfolder + "/submission.csv"):
                submissions[submission_name] = DataScienceSubmissionCard.load(subfolder)
        return submissions

    def reset(self) -> tuple[dict[DSObsKey, str], dict[DSObsKey, ...]]:
        self.done = False
        self.obs = {
            DSObsKey.TASK_DESCRIPTION: self.get_task_description(),
            DSObsKey.DATA_DESCRIPTION: self.get_data_description(),
            DSObsKey.SUMMARIZED_METRIC_DESCRIPTION: self.get_metric_description(),
            DSObsKey.SUBMISSION_LIST: [],
            DSObsKey.CURRENT_SUBMISSION: None,
            DSObsKey.SENT_SUBMISSION_NAMES: [],
        }

        assert self.has_table_input or self.has_img_input or self.has_txt_input
        if not self.has_table_target and not self.has_img_target and not self.has_txt_target:
            raise ValueError(f"No target - check in\n\t- {self.workspace_path}\n\t- {self.get_src_path()}")

        self.top_stage = TopDatascienceStage(
            has_table_input=self.has_table_input,
            has_regression_target=self.has_regression_target,
            has_classification_target=self.has_classification_target,
            has_img_input=self.has_img_input,
            has_txt_input=self.has_txt_input,
            has_table_target=self.has_table_target,
            has_img_target=self.has_img_target,
            has_txt_target=self.has_txt_target
        )

        self.start_time = time.time()
        self.step_num = 0

        # Create workspace directory
        os.makedirs(self.workspace_path, exist_ok=True)

        self.current_submission = None

        info: dict[DSObsKey, ...] = {DSObsKey.AVAILABLE_ACTIONS: self.get_available_actions()}

        # Copy the useful scripts in the workspace
        self.retrieve_metric_script()
        self.retrieve_submission_format_scripts()

        if os.path.exists(self.get_src_target_col_classification_transform_path()):
            shutil.copyfile(src=self.get_src_target_col_classification_transform_path(),
                            dst=self.get_dst_col_classification_transform_path())
        if os.path.exists(self.get_src_train_tab_target_path()):
            shutil.copyfile(src=self.get_src_train_tab_target_path(), dst=self.get_dst_train_tab_target_path())
        if os.path.exists(self.get_src_tab_target_transform_path()):
            shutil.copyfile(src=self.get_src_tab_target_transform_path(), dst=self.get_dst_tab_target_transform_path())
        if os.path.exists(self.get_src_img_target_transform_path()):
            shutil.copyfile(src=self.get_src_img_target_transform_path(), dst=self.get_dst_img_target_transform_path())
        if os.path.exists(self.get_src_txt_target_transform_path()):
            shutil.copyfile(src=self.get_src_txt_target_transform_path(), dst=self.get_dst_txt_target_transform_path())
        if os.path.exists(self.get_src_sample_submission_path()):
            shutil.copyfile(src=self.get_src_sample_submission_path(), dst=self.get_dst_sample_submission_path())
        return self.obs, info

    def resume(self) -> tuple[dict[DSObsKey, str], float, bool, dict[DSObsKey, ...]]:
        """ Resume from previous run """
        new_submission_path = os.path.join(self.workspace_path, DS_SUBMISSIONS_DIRNAME, NEW_SUBMISSION_DIRNAME)
        if os.path.exists(new_submission_path):
            shutil.rmtree(new_submission_path)

        self.load_resume_checkpoint()
        self.current_submission = None
        self.start_time = time.time()

        reward = self.get_reward()
        info: dict[DSObsKey, ...] = {}
        self.obs[DSObsKey.SUBMISSION_LIST] = list(self.get_submissions().values())
        if (time.time() - self.start_time) >= self.max_exec_time:
            print(f"Exhausted maximum execution time {self.max_exec_time}")
            self.done = True
            info[DSObsKey.AVAILABLE_ACTIONS] = []
        else:
            info[DSObsKey.AVAILABLE_ACTIONS] = self.get_available_actions()
        return self.obs, reward, self.done, info

    def save_resume_checkpoint(self) -> None:
        """ Save env stage """
        resume_checkpoint = ResumeCheckpoint(
            obs=self.obs,
            top_stage=self.top_stage,
            max_exec_time=self.max_exec_time - (time.time() - self.start_time),
            step_num=self.step_num
        )
        save_w_pickle(obj=resume_checkpoint, path=self.workspace_path, filename=FileMap.RESUME_CHECKPOINT_FILE.value)

    def load_resume_checkpoint(self) -> None:
        """ Load env stage """
        if os.path.exists(os.path.join(self.workspace_path, FileMap.RESUME_CHECKPOINT_FILE.value)):
            resume_checkpoint = load_w_pickle(path=self.workspace_path, filename=FileMap.RESUME_CHECKPOINT_FILE.value)
            self.max_exec_time = resume_checkpoint.max_exec_time
            self.obs: dict[DSObsKey, ...] = resume_checkpoint.obs
            self.top_stage = resume_checkpoint.top_stage
            self.step_num = resume_checkpoint.step_num
        else:
            raise FileNotFoundError(
                f"File not found: {os.path.join(self.workspace_path, FileMap.RESUME_CHECKPOINT_FILE.value)}."
                f"You can try a normal run instead of resume run"
            )

    def get_available_actions(self) -> list[str]:
        all_available_actions = [stage.name.value for stage in self.top_stage.get_available_stages()]
        actions_to_filter_out = [DataScienceStageNames.ADD_HYPERPARAMETERS, DataScienceStageNames.BAG_SUBMISSIONS]
        actions_to_filter_out = [action.value for action in actions_to_filter_out]
        if len(self.get_submissions()) >= int(
                os.getenv('BLEND_AFTER_N', '500')) and DataScienceStageNames.BLEND_SUBMISSIONS in all_available_actions:
            print('AUTOMATIC TRIGGER OF BLENDING!')
            return [DataScienceStageNames.BLEND_SUBMISSIONS]
        elif len(self.get_submissions()) < 2:
            actions_to_filter_out.append(DataScienceStageNames.BLEND_SUBMISSIONS)
        return [action for action in all_available_actions if action not in actions_to_filter_out]

    def step(self, action: DSAction, retrial_chat_completion_time: float) \
            -> tuple[dict[DSObsKey, str], float, bool, dict[DSObsKey, ...]]:
        self.max_exec_time += retrial_chat_completion_time
        self.top_stage.update(stage_name=action.stage_name.value)
        self.step_num += 1
        self.action_dependent_step(action=action)
        reward = self.get_reward()
        info: dict[DSObsKey, ...] = {}
        self.obs[DSObsKey.SUBMISSION_LIST] = list(self.get_submissions().values())
        if (time.time() - self.start_time) >= self.max_exec_time:
            print(f"Exhausted maximum execution time {self.max_exec_time}")
            self.done = True
            info[DSObsKey.AVAILABLE_ACTIONS] = []
        else:
            info[DSObsKey.AVAILABLE_ACTIONS] = self.get_available_actions()
        return self.obs, reward, self.done, info

    def get_src_path(self) -> str:
        return os.path.join(self.prepared_setup_dir, self.task_id, self.prepared_version)

    def get_dst_metric_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.METRIC_SCRIPT.value)

    def get_dst_col_classification_transform_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS.value)

    def get_dst_train_tab_target_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.TRAIN_TABULAR_TARGET.value)

    def get_dst_tab_target_transform_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.TAB_TARGETS_TRANSFORM.value)

    def get_dst_img_target_transform_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.IMG_TARGETS_TRANSFORM.value)

    def get_dst_txt_target_transform_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.TXT_TARGETS_TRANSFORM.value)

    def get_dst_submission_format_path(self, alt: bool) -> str:
        if alt:
            sub_format_script = FileMap.SUBMISSION_FORMAT_ALT_SCRIPT
        else:
            sub_format_script = FileMap.SUBMISSION_FORMAT_SCRIPT

        return os.path.join(self.workspace_path, sub_format_script.value)

    def get_dst_sample_submission_path(self) -> str:
        return os.path.join(self.workspace_path, FileMap.SAMPLE_SUBMISSION_FILE.value)

    def get_element_src_path(self, filemap_key: FileMap.name) -> str:
        """ Get the path of an element from source folder """
        return os.path.join(self.get_src_path(), filemap_key.value)

    def get_src_submission_format_path(self, alt: bool):
        if alt:
            filemap_key = FileMap.SUBMISSION_FORMAT_ALT_SCRIPT
        else:
            filemap_key = FileMap.SUBMISSION_FORMAT_SCRIPT
        return self.get_element_src_path(filemap_key=filemap_key)

    get_src_metric_path = partialmethod(get_element_src_path, FileMap.METRIC_SCRIPT)
    get_src_target_col_classification_transform_path = partialmethod(get_element_src_path,
                                                                     FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS)
    get_src_train_tab_target_path = partialmethod(get_element_src_path, FileMap.TRAIN_TABULAR_TARGET)

    # get_src_tab_target_transform_path = partialmethod(get_element_src_path, FileMap.TAB_TARGETS_TRANSFORM)
    def get_src_tab_target_transform_path(self):
        workdir_tab_target_transform = self.get_element_src_path(FileMap.TAB_TARGETS_TRANSFORM)
        if workdir_tab_target_transform is None:
            # If the file doesn't exist in the original location return None to be safe
            return None
        # tab_target_train transform is a static file so we can copy the newest version directly
        original_file = PROJECT_ROOT / "src/agent/prompts/templates/data_preprocessing/data_map/code_template/transform/tab_target_train.py"
        assert original_file.exists(), "Cannot find transform/tab_target_train.py, has the project structure changed?"
        return original_file

    get_src_img_target_transform_path = partialmethod(get_element_src_path, FileMap.IMG_TARGETS_TRANSFORM)
    get_src_txt_target_transform_path = partialmethod(get_element_src_path, FileMap.TXT_TARGETS_TRANSFORM)
    get_src_sample_submission_path = partialmethod(get_element_src_path, FileMap.SAMPLE_SUBMISSION_PATH)

    def get_description_from_src(self, filemap_key: FileMap.name) -> str:
        path = self.get_element_src_path(filemap_key=filemap_key)
        with open(path) as f:
            description = "".join(f.readlines())

        return description

    get_task_description = partialmethod(get_description_from_src, filemap_key=FileMap.TASK_DESCRIPTION)
    get_data_description = partialmethod(get_description_from_src, filemap_key=FileMap.DATA_DESCRIPTION)
    get_metric_description = partialmethod(get_description_from_src, filemap_key=FileMap.METRIC_DESCRIPTION)

    def get_reward(self) -> float:
        if self.top_stage.submit.is_fully_over():
            return int(self.obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY])
        return 0

    def create_dnn_submission_state(self) -> DNNSubmissionState:
        """ Create a submission made of DNNs """
        return DNNSubmissionState(
            table_fe=TableFEStatus(
                is_necessary=self.has_table_input, is_available=self.has_table_input, is_done=False, is_pending=False
            ),
            table_model=TableModelStatus(
                is_necessary=self.has_table_input, is_available=self.has_table_input, is_done=False, is_pending=False
            ),
            table_embed=TableEmbedStatus(
                is_necessary=self.has_table_input, is_available=self.has_table_input, is_done=False, is_pending=False
            ),
            img_embed=ImgEmbedStatus(
                is_necessary=self.has_img_input, is_available=self.has_img_input, is_done=False, is_pending=False
            ),
            train_img_transform=TrainImgTransformStatus(
                is_necessary=self.has_img_input, is_available=self.has_img_input, is_done=False, is_pending=False
            ),
            test_img_transform=TestImgTransformStatus(
                is_necessary=self.has_img_input, is_available=self.has_img_input, is_done=False, is_pending=False
            ),
            img_model=ImgModelStatus(
                is_necessary=self.has_img_input, is_available=self.has_img_input, is_done=False, is_pending=False
            ),
            txt_embed=TxtEmbedStatus(
                is_necessary=self.has_txt_input, is_available=self.has_txt_input, is_done=False, is_pending=False),
            table_head=TableHeadStatus(
                is_necessary=self.has_table_target, is_available=self.has_table_target, is_done=False, is_pending=False
            ),
            regression_target_transform=RegeressionTargetTransformStatus(
                is_necessary=self.has_regression_target, is_available=self.has_regression_target, is_done=False,
                is_pending=False
            ),
            class_imbalance=ClassImbalanceStatus(
                is_necessary=self.has_classification_target, is_available=self.has_classification_target, is_done=False,
                is_pending=False
            ),
            img_head=ImgHeadStatus(
                is_necessary=self.has_img_target, is_available=self.has_img_target, is_done=False, is_pending=False
            ),
            txt_head=TxtHeadStatus(
                is_necessary=self.has_txt_target, is_available=self.has_txt_target, is_done=False, is_pending=False
            ),
            bag=BagStatus(is_necessary=False, is_available=True, is_done=False, is_pending=False),
            blend=BlendStatus(is_necessary=False, is_available=True, is_done=False, is_pending=False)
        )

    def cp_workdir_to_submission(self, basename: str, replacements: dict[str, str] | None,
                                 not_exist_ok: bool = False) -> None:
        """ Move file from workdir to current submission dir and replace patterns using the `replacements` dict """
        src = os.path.join(self.workspace_path, basename)
        if not os.path.exists(src) and not_exist_ok:
            return
        dst_path = os.path.join(self.current_submission.path, basename)

        if not os.path.exists(dst_path):
            shutil.copy(src, dst_path)
        if replacements is not None:
            replace_in_file(file_path=dst_path, replacements=replacements)

    def action_dependent_step(self, action: DSAction) -> None:
        """ Update the env based on the selected action """

        if action.stage_name == DataScienceStageNames.ADOPT_DNN_FRAMEWORK:
            return

        if action.stage_name == DataScienceStageNames.ADD_SUBMISSION:
            self.add_submission()
            return

        if action.stage_name == DataScienceStageNames.ADOPT_CLASSICAL_ML_FRAMEWORK:
            os.makedirs(os.path.join(self.workspace_path, "data"), exist_ok=True)
            return

        if action.stage_name == DataScienceStageNames.ADD_CLASSICAL_TAB_SUBMISSION:
            # Create a classical ML submission and mark it as current
            self.add_submission()

            submission = self.load_current_submission()
            submission.submission_state = ClassicalTabSubmissionState(
                table_fe=TableFEStatus(is_necessary=self.has_table_input,
                                       is_available=self.has_table_input, is_done=False, is_pending=False)
            )
            self.current_submission = submission
            self.obs[DSObsKey.CURRENT_SUBMISSION] = self.current_submission
            submission.save()
            return

        if action.stage_name == DataScienceStageNames.CLASSICAL_TAB_FE:
            # State of current submission should be updated as there is feature engineering
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, ClassicalTabSubmissionState), type(
                submission.submission_state)
            submission.submission_state.table_fe.is_done = False

            # extract summary
            submission.submission_state.table_fe.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK].replace("\n", "\n    ")
            model_out_put, num_iter = check_accuracy_classical_tab_fe(code_block, self.workspace_path)
            self.obs[DSObsKey.CURRENT_SUBMISSION_PERF] = model_out_put
            if num_iter == self.top_stage.classical_ml_route._target_n_times:
                submission.submission_state.table_fe.is_done = True
            submission.save()

            # --- update the code for table_fe
            self.cp_workdir_to_submission(basename="tab_fe.py", replacements={"# @TABLE_FE_CODE@": code_block})

            return

        if action.stage_name == DataScienceStageNames.CLASSICAL_TAB_MODEL:
            raise NotImplementedError()
        if action.stage_name == DataScienceStageNames.CLASSICAL_TAB_HYP:
            raise NotImplementedError()
        if action.stage_name == DataScienceStageNames.CLASSICAL_TAB_TRAIN:
            raise NotImplementedError()

        if action.stage_name == DataScienceStageNames.ADD_DNN_SUBMISSION:
            # Create a DNN submission and mark it as current
            submission = self.load_current_submission()
            submission.submission_state = self.create_dnn_submission_state()
            self.current_submission = submission
            self.obs[DSObsKey.CURRENT_SUBMISSION] = self.current_submission
            submission.save()

            # --- add training code + metric
            self.cp_workdir_to_submission(basename=FileMap.SOLVE_SCRIPT.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.SOLVE_COMMON_UTILS.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.TRAIN_UTILS.value,
                                          replacements={"@ROOT_DS_DATA_PATH@": self.get_src_path()})
            self.cp_workdir_to_submission(basename="solve_params.py", replacements=None)
            self.cp_workdir_to_submission(basename="create_blend_dataset.py", replacements=None)
            self.cp_workdir_to_submission(basename="blend_params.py", replacements=None)
            return

        if action.stage_name == DataScienceStageNames.CLASS_IMBALANCE:
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.class_imbalance.is_done = True
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(
                basename="class_imbalance.py", replacements={"# @CLASS_IMBALANCE_CODE@": code_block}
            )
            return

        if action.stage_name == DataScienceStageNames.DNN_TAB_EMBEDDING:
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.table_embed.is_pending = True
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()
            return

        if action.stage_name == DataScienceStageNames.TAB_FEATURE_ENGINEERING:
            # State of current submission should be updated as there is feature engineering
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.table_fe.is_done = True
            # extract summary
            submission.submission_state.table_fe.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for table_fe
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(basename="tab_fe.py", replacements={"# @TABLE_FE_CODE@": code_block})
            return

        if action.stage_name == DataScienceStageNames.TAB_EMBED_PREPROCESSED:
            # State of current submission should be updated as there is modeling
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.table_model.is_done = True
            submission.submission_state.table_embed.is_done = True
            # extract summary
            submission.submission_state.table_model.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for table_fe
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(
                basename="tab_embed.py", replacements={"# @TABLE_EMBED_PREPROCESSED_CODE@": code_block}
            )
            return

        if action.stage_name == DataScienceStageNames.IMAGE_EMBEDDING:
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.img_embed.is_pending = True
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()
            return

        if action.stage_name == DataScienceStageNames.TRAIN_IMAGE_TRANSFORM:
            # State of current submission should be updated as there is modeling
            submission = self.load_current_submission()
            if not isinstance(submission.submission_state, DNNSubmissionState):
                raise ValueError(f"{type(submission)}, {type(submission.submission_state)}")
            submission.submission_state.train_img_transform.is_done = True
            # extract summary
            submission.submission_state.train_img_transform.specific_description = action.hyps[
                DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for image transform
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(
                basename="img_transform.py", replacements={"# @TRAIN_IMG_TRANSFORM_CODE@": code_block}
            )
            return

        if action.stage_name == DataScienceStageNames.TEST_IMAGE_TRANSFORM:
            # State of current submission should be updated as there is modeling
            submission = self.load_current_submission()
            if not isinstance(submission.submission_state, DNNSubmissionState):
                raise ValueError(f"{type(submission)}, {type(submission.submission_state)}")
            submission.submission_state.test_img_transform.is_done = True
            # extract summary
            submission.submission_state.test_img_transform.specific_description = action.hyps[
                DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for image transform
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(
                basename="img_transform.py", replacements={"# @TEST_IMG_TRANSFORM_CODE@": code_block}
            )
            return

        if action.stage_name == DataScienceStageNames.IMAGE_MODELLING:
            # State of current submission should be updated as there is modeling
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.img_model.is_done = True
            submission.submission_state.img_embed.is_done = True
            # extract summary
            submission.submission_state.img_model.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for image model
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(basename="img_embed.py", replacements={"# @IMG_EMBED_CODE@": code_block})
            return

        if action.stage_name == DataScienceStageNames.TEXT_EMBEDDING:
            # State of current submission should be updated as there is modeling
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), (
                type(submission), type(submission.submission_state))
            submission.submission_state.txt_embed.is_done = True
            # extract summary
            submission.submission_state.txt_embed.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for image  embedding
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(basename="txt_embed.py", replacements={"# @TXT_EMBED_CODE@": code_block})
            return

        if action.stage_name == DataScienceStageNames.TABLE_REGRESSION_TARGET_TRANSFORM:
            # State of current submission should be updated as there is tabular head
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.regression_target_transform.is_done = True
            # extract summary
            submission.submission_state.regression_target_transform.specific_description = action.hyps[
                DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for tab_taeg
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(
                basename=FileMap.TAB_REGRESSION_TARGETS_TRANSFORM.value,
                replacements={"# @TAB_REGRESSION_TARGET_CODE@": code_block}
            )
            return

        if action.stage_name == DataScienceStageNames.TABLE_HEAD:
            # State of current submission should be updated as there is tabular head
            submission = self.load_current_submission()
            assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            submission.submission_state.table_head.is_done = True
            # extract summary
            submission.submission_state.table_head.specific_description = action.hyps[DSActionHypKeys.SUMMARY_STEP]
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            # --- update the code for tab_head
            code_block = action.hyps[DSActionHypKeys.CODE_BLANK]
            self.cp_workdir_to_submission(basename="tab_head.py", replacements={"# @TAB_HEAD_CODE@": code_block})
            return

        if action.stage_name == DataScienceStageNames.IMAGE_HEAD:
            raise NotImplementedError()
        if action.stage_name == DataScienceStageNames.TEXT_HEAD:
            raise NotImplementedError()

        if action.stage_name == DataScienceStageNames.ADD_TRAIN_CODE:
            # State of current submission should be updated as there is a training code
            submission = self.load_current_submission()
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()

            return

        if action.stage_name == DataScienceStageNames.ADD_HYPERPARAMETERS:
            # State of current submission should be updated as there are hyperparameters
            submission = self.load_current_submission()
            self.obs[DSObsKey.CURRENT_SUBMISSION] = submission
            submission.save()
            return

        if action.stage_name == DataScienceStageNames.TRAIN_SUBMISSION:
            # Train the model and compute predictions

            # summarize the submission and change its name, add to the list of submissions
            submission = self.load_current_submission()
            submission.submission_state.summary = action.hyps.get(DSActionHypKeys.SUBMISSION_SUMMARY)

            python_exe = get_path_to_ds_python()
            script = FileMap.SOLVE_SCRIPT.value
            error_log = FileMap.SOLVE_ERROR_LOG.value
            output_log = FileMap.SOLVE_OUTPUT_LOG.value
            remaining_time = self.max_exec_time - (time.time() - self.start_time)
            training_time_limit = min(remaining_time, float(os.getenv("MAX_TIME_PER_SUBMISSION", 18 * 3600)))
            cmd = (f"cd {submission.path} && {python_exe} {script} --max_total_runtime={training_time_limit} "
                   f"2> {error_log} > {output_log}")

            print(cmd, flush=True)
            with open(f"{submission.path}/train_command.txt", "w") as f:
                f.write(cmd)
            os.system(cmd)

            # Check if the code ran properly,
            # i.e. created a submission.csv or alt, and update the submission summary with the validation loss
            submission_path = os.path.join(submission.path, FileMap.SUBMISSION_FILE.value)
            submission_path_alt = os.path.join(submission.path, FileMap.SUBMISSION_ALT_FILE.value)

            if os.path.exists(submission_path) or os.path.exists(submission_path_alt):
                validation_score_path = os.path.join(submission.path, FileMap.VALIDATION_LOSS_FEEDBACK.value)
                assert os.path.exists(validation_score_path), validation_score_path
                with open(validation_score_path) as f:
                    validation_score_message = f.read()

                submission.submission_state.summary += "\n" + validation_score_message

            submission.update_given_name(
                new_name=action.hyps.get(DSActionHypKeys.SUBMISSION_NAME, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            )
            self.obs[DSObsKey.CURRENT_SUBMISSION] = None
            submission.save()

            if self.terminate_after_training:
                self.done = True

            # save env status
            self.save_resume_checkpoint()

            return

        if action.stage_name == DataScienceStageNames.BAG_SUBMISSIONS:
            raise NotImplementedError()

        if action.stage_name == DataScienceStageNames.BLEND_SUBMISSIONS:
            name = NEW_BLEND_DIRNAME
            submission = DataScienceSubmissionCard(root_path=self.submissions_path, name=name, submission_state=None)
            submission.create()
            self.current_submission = submission

            # Add auxiliary files
            self.cp_workdir_to_submission(basename=FileMap.METRIC_SCRIPT.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.SUBMISSION_FORMAT_SCRIPT.value, replacements=None)
            self.cp_workdir_to_submission(
                basename=FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value, replacements=None, not_exist_ok=True
            )
            self.cp_workdir_to_submission(basename=FileMap.TRAIN_TABULAR_TARGET.value, replacements=None,
                                          not_exist_ok=True)
            self.cp_workdir_to_submission(basename=FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS.value,
                                          replacements=None,
                                          not_exist_ok=True)
            self.cp_workdir_to_submission(basename=FileMap.TAB_TARGETS_TRANSFORM.value, replacements=None,
                                          not_exist_ok=True)
            self.cp_workdir_to_submission(basename=FileMap.IMG_TARGETS_TRANSFORM.value, replacements=None,
                                          not_exist_ok=True)
            self.cp_workdir_to_submission(basename=FileMap.TXT_TARGETS_TRANSFORM.value, replacements=None,
                                          not_exist_ok=True)
            self.cp_workdir_to_submission(basename=FileMap.BLEND_SCRIPT.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.SOLVE_SCRIPT.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.SOLVE_COMMON_UTILS.value, replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.TRAIN_UTILS.value, replacements=None)
            self.cp_workdir_to_submission(basename="solve_params.py", replacements=None)
            self.cp_workdir_to_submission(basename="create_blend_dataset.py", replacements=None)
            self.cp_workdir_to_submission(basename="blend_params.py", replacements=None)
            self.cp_workdir_to_submission(basename="map_dataset.py", replacements=None)
            self.cp_workdir_to_submission(basename=FileMap.SAMPLE_SUBMISSION_FILE.value, replacements=None)

            submission_names = action.hyps[DSActionHypKeys.SUBMISSIONS_LIST]
            submission_names = submission_names.replace(" ", "").replace("[", "").replace("]", "").split(",")
            submissions_to_blend = []
            for submission_name in submission_names:
                submissions_to_blend.append(
                    os.path.abspath(os.path.join(self.workspace_path, f'submissions/{submission_name}')))

            # submission = self.load_current_submission()

            # assert isinstance(submission.submission_state, DNNSubmissionState), type(submission.submission_state)
            # submission.submission_state.blend.is_done = True
            # Run the blending code to generate final submission
            python_exe = get_path_to_ds_python()
            script = FileMap.BLEND_SCRIPT.value
            error_log = FileMap.BLEND_ERROR_LOG.value
            output_log = FileMap.BLEND_OUTPUT_LOG.value
            for i in range(len(submissions_to_blend)):
                print(
                    f"[{i + 1}/{len(submissions_to_blend)}] "
                    f"Creating blend datasets of submission {submissions_to_blend[i]} ...",
                    flush=True
                )
                os.system(f'cd {submissions_to_blend[i]} && {python_exe} create_blend_dataset.py '
                          f'2> create_blend_dataset_error.log > create_blend_dataset_output.log')
            os.system(f'cp -r {submissions_to_blend[0]}/tab_head.py {submission.path}')
            cmd = f"cd {submission.path} && {python_exe} {script} --submissions {' '.join(submissions_to_blend)}"
            cmd_redirect = f"{cmd} 2> {error_log} > {output_log}"
            print(cmd_redirect, flush=True)
            with open(f"{submission.path}/{FileMap.BLEND_COMMAND_TXT.value}", "w") as f:
                f.write(cmd_redirect)
            os.system(cmd_redirect)
            submission.update_given_name(
                new_name=action.hyps.get(DSActionHypKeys.SUBMISSION_NAME,
                                         f"blend_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
            )
            # submission.submission_state.summary = action.hyps.get(DSActionHypKeys.SUBMISSION_SUMMARY)
            self.obs[DSObsKey.CURRENT_SUBMISSION] = None
            submission.save()

            # The blend is run after all the other submissions so we directly submit it to kaggle
            obs = self.submit_to_kaggle(
                submission_file_path=os.path.join(submission.path, FileMap.BLEND_SUBMISSION_FILE.value),
                submission_name=submission.name
            )
            self.obs.update(obs)
            if obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY]:
                self.obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY].append(submission.name)
            self.done = True
            return

        if action.stage_name == DataScienceStageNames.SEND_SUBMISSION:
            submission_name = action.hyps[DSActionHypKeys.SUBMISSION_NAME]
            submission_file_paths = [
                os.path.join(
                    self.submissions_path,
                    submission_name,
                    FileMap.SUBMISSION_FILE.value
                ),
                os.path.join(
                    self.submissions_path,
                    submission_name,
                    FileMap.SUBMISSION_ALT_FILE.value
                )
            ]
            submission_obs = [
                self.submit_to_kaggle(p, submission_name) for p in submission_file_paths if os.path.exists(p)
            ]
            if not submission_obs:
                raise FileNotFoundError(f"Neither submission file is present: {submission_file_paths}")

            successful_submissions = [obs for obs in submission_obs if obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY]]
            if not successful_submissions:
                self.obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY] = False
            else:
                self.obs[DSObsKey.SENT_SUBMISSION_NAMES].append(submission_name)

            self.done = True
            return

        else:
            raise RuntimeError(action)

    def submit_to_kaggle(self, submission_file_path: str, submission_name: str, phase="public") -> dict[DSObsKey, ...]:
        """Submit specified file to kaggle via the API
        also attempts to join the competition via selenium"""
        from requests.exceptions import HTTPError
        from kaggle.api.kaggle_api_extended import KaggleApi

        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        assert phase in ("public", "private")
        assert phase == "public", "phase=private does not work unless the request.get has a kaggle login cookie"

        obs = {}
        try:
            try:
                # This can get stuck when the proxy blocks it -- set KAGGLE_PROXY
                submit_result = kaggle_api.competition_submit(
                    file_name=submission_file_path,
                    message=submission_name,
                    competition=self.task_id
                )
            except HTTPError as e:
                body = e.response.json()
                accept_rules_message = "You must accept the rules for this competition to perform this action."
                if body["code"] == 403 and body["message"] == accept_rules_message:
                    # This indicates the competition has not been joined, so try to join it with selenium
                    from agent.tools.fetch_tool import FetchTool
                    fetch_tool = FetchTool(
                        task_url=f"https://www.kaggle.com/competitions/{self.task_id}",
                        user_details="./third_party/data_preprocessing/kaggle_login_details.json",
                        # Hopefully these are irrelevant for joining
                        workspace_path=".",
                        raw_data_dir=".",
                        is_local_task=self.is_local_task
                    )
                    fetch_tool.join_competition(fetch_tool.task_url)

                    submit_result = kaggle_api.competition_submit(
                        file_name=submission_file_path,
                        message=submission_name,
                        competition=self.task_id
                    )
                else:
                    raise

            obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY] = True
        except:
            obs[DSObsKey.SUBMISSION_SENT_SUCCESSFULLY] = False

        return obs

    def load_current_submission(self) -> DataScienceSubmissionCard:
        path = DataScienceSubmissionCard.get_path(
            root_path=os.path.join(self.workspace_path, DS_SUBMISSIONS_DIRNAME), name=self.current_submission.name
        )
        self.current_submission = DataScienceSubmissionCard.load(path)
        return self.current_submission

    @property
    def table_input_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_TABULAR_INPUT.value)

    @staticmethod
    def generic_table_check(csv_path: str) -> bool:
        """ Check whether csv exists and whether it is empty or not """
        if not os.path.exists(csv_path):
            return False
        data = pd.read_csv(csv_path, index_col=ID_COLUMN_NAME)
        if len(data.columns) == 0:
            return False
        return True

    @property
    def has_table_input(self) -> bool:
        return self.generic_table_check(csv_path=self.table_input_path)

    @property
    def has_regression_target(self) -> bool:
        """ Whether the task involves a regression target """
        if not os.path.exists(self.table_target_path):
            return False
        target_table_df = pd.read_csv(self.table_target_path, index_col=ID_COLUMN_NAME)
        #  Check if one target column ends with "_regression"
        class_names_columns_regression = [col for col in target_table_df.columns if col.endswith('_regression')]
        return len(class_names_columns_regression) > 0

    @property
    def has_classification_target(self) -> bool:
        """ Whether the task involves a regression target """
        if not os.path.exists(self.table_target_path):
            return False
        target_table_df = pd.read_csv(self.table_target_path, index_col=ID_COLUMN_NAME)
        #  Check if one target column ends with "_regression"
        class_names_columns_regression = [col for col in target_table_df.columns if col.endswith('_classification')]
        return len(class_names_columns_regression) > 0

    @property
    def image_input_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_IMAGE_INPUT.value)

    @property
    def has_img_input(self) -> bool:
        return self.generic_table_check(csv_path=self.image_input_path)

    @property
    def text_input_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_TEXT_INPUT.value)

    @property
    def has_txt_input(self) -> bool:
        return self.generic_table_check(csv_path=self.text_input_path)

    @property
    def table_target_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_TABULAR_TARGET.value)

    @property
    def has_table_target(self) -> bool:
        return self.generic_table_check(csv_path=self.table_target_path)

    @property
    def image_target_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_IMAGE_TARGET.value)

    @property
    def has_img_target(self) -> bool:
        return self.generic_table_check(csv_path=self.image_target_path)

    @property
    def text_target_path(self) -> str:
        return os.path.join(self.get_src_path(), FileMap.TRAIN_TEXT_TARGET.value)

    @property
    def has_txt_target(self) -> bool:
        return self.generic_table_check(csv_path=self.text_target_path)

    def add_submission(self) -> None:
        """ Create a submission and copy auxiliary files"""
        name = NEW_SUBMISSION_DIRNAME
        submission = DataScienceSubmissionCard(root_path=self.submissions_path, name=name, submission_state=None)
        submission.create()
        self.current_submission = submission

        # Add auxiliary files
        self.cp_workdir_to_submission(basename=FileMap.METRIC_SCRIPT.value, replacements=None)
        self.cp_workdir_to_submission(basename=FileMap.SUBMISSION_FORMAT_SCRIPT.value, replacements=None)
        self.cp_workdir_to_submission(
            basename=FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value, replacements=None, not_exist_ok=True
        )
        self.cp_workdir_to_submission(basename=FileMap.SAMPLE_SUBMISSION_FILE.value, replacements=None)
        self.cp_workdir_to_submission(basename=FileMap.TRAIN_TABULAR_TARGET.value, replacements=None,
                                      not_exist_ok=True)
        self.cp_workdir_to_submission(basename=FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS.value,
                                      replacements=None,
                                      not_exist_ok=True)
        self.cp_workdir_to_submission(basename=FileMap.TAB_TARGETS_TRANSFORM.value, replacements=None,
                                      not_exist_ok=True)
        self.cp_workdir_to_submission(basename=FileMap.IMG_TARGETS_TRANSFORM.value, replacements=None,
                                      not_exist_ok=True)
        self.cp_workdir_to_submission(basename=FileMap.TXT_TARGETS_TRANSFORM.value, replacements=None,
                                      not_exist_ok=True)
        self.cp_workdir_to_submission(basename="map_dataset.py", replacements=None)

        return

    @staticmethod
    def unwrap_warning_redirect(file_path: str) -> None:
        """
        Replace the file in file path by the same code without warnings redirection
        """

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if "warnings.showwarning = write_warning_to_file" in line:
                continue
            new_lines.append(line)

        # Remove what's in the outer try except
        def deindent(body: str) -> str:
            if body.startswith("    "):
                body = body[4:]
            elif body.startswith("\t"):
                body = body[1:]
            return body

        file_content = "".join(new_lines)
        start_pattern = "\ntry:"
        start = file_content.find(start_pattern)
        end_pattern = "\nexcept"
        end = file_content.rfind(end_pattern)
        file_head = file_content[:start] + "\n"
        file_body_split = (file_content[start + len(start_pattern):end] + "\n").split("\n")
        file_body = "\n".join(list(map(lambda x: deindent(body=x), file_body_split)))

        file_content = file_head + file_body
        with open(file_path, "w") as f:
            f.write(file_content)

    def retrieve_metric_script(self) -> None:
        # Copy the metric from the setup to the workdir
        shutil.copyfile(src=self.get_src_metric_path(), dst=self.get_dst_metric_path())
        self.unwrap_warning_redirect(file_path=self.get_dst_metric_path())

    def retrieve_submission_format_scripts(self) -> None:
        # Copy the submission formatting to the workdir
        n_exists = 0
        for alt in [False, True]:
            src_path = self.get_src_submission_format_path(alt=alt)
            if not os.path.exists(src_path):
                continue
            dst_path = self.get_dst_submission_format_path(alt=alt)
            shutil.copyfile(src=src_path, dst=dst_path)
            self.unwrap_warning_redirect(file_path=dst_path)
            n_exists += 1
        assert n_exists > 0
