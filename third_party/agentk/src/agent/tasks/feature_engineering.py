from __future__ import annotations

import json
import os
import pathlib
import shutil

from agent.memory import MemKey
from agent.prompts.builder import BASE_TEMPLATE_PATH
from agent.tasks import ActionSpace
from agent.tasks import Task
from agent.tasks.datascience_task.utils import FileMap, replace_in_file
from agent.utils import pylogger
from agent.utils.utils import ListableEnum
from third_party.data_science.env import DSAction, DSActionHypKeys

log = pylogger.get_pylogger(__name__)


class FeatureEngineeringStageNames(str, ListableEnum):
    START = "start"
    CLASSICAL_TAB_FE = "Add feature engineering"
    MODEL_TRAINING = "Train model"
    SELECT_BEST_MODEL = "Select best model"
    GENERATE_TAB_COLUMN_TYPES = "create_column_types"
    TERMINATE = "Terminate"


class FeatureEngineering(Task):
    def __init__(
            self,
            task_id: str,
            prepared_setup_dir: str,
            prepared_version: str,
            exp_id: str,
            workspace_path: str,
            name: str,
            version: str,
            subtask: str | None = None,
            **kwargs,
    ):
        """
        Args:
            task_id: ID of the task kit to be completed.
            prepared_version: ID of the data preparation applied to the task data
            exp_id: automatically created from task_id, prepared_version and timestamp
            workspace_path: Path to the new workspace to be created.
        """
        if subtask is None:
            subtask = task_id
        self.task_id = task_id
        self.prepared_setup_dir = prepared_setup_dir
        self.prepared_version = prepared_version
        self.exp_id = exp_id
        self.workspace_path = workspace_path
        self.action_space = ActionSpace.DISCRETE
        self.done = False
        self.step_num = 1  # Number of fe iterations required
        self.stages = [member for member in FeatureEngineeringStageNames]
        self.active_stage = None
        self.fe_code = ""
        self.performance = {}
        super().__init__(name=name, subtask=subtask, version=version, **kwargs)

    def template_code_reset(self) -> None:
        """Create or moved the templates that are task-dependant."""
        os.makedirs(self.templates_path, exist_ok=True)
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "trials"), exist_ok=True)

        current_dir = pathlib.Path(__file__).parent.resolve()
        desired_dir = str(os.path.join(current_dir.parent, "prompts", "templates", "feature_engineering"))
        replacements = {
            "@ROOT_DS_DATA_PATH@": self.get_src_path(),
            "@WORKSPACE@": self.workspace_path,
            "@TASK_ID@": self.task_id
        }

        code_templates = [
            "classical_tab_fe_code_template.py",
            "column_types/code_template_column_types.py",
            "training/classical_tab_training_code_template.py",
            "select_best_model/select_best_performance_template.py"
        ]

        def get_template_dst(path: str) -> str:
            """Get destination path of the template."""
            p, ext = os.path.splitext(path)
            if ext != ".jinja":
                path = p + ".jinja"
            return os.path.join(self.templates_path, os.path.basename(path))

        def get_workspace_dst(path: str) -> str:
            """Get destination path of the code blank."""
            p, ext = os.path.splitext(path)
            if ext != ".py":
                path = p + ".py"
            return os.path.join(self.workspace_path, os.path.basename(path))

        for code_template in code_templates:
            code_template = desired_dir + "/" + code_template
            dst_path = get_template_dst(path=code_template)
            shutil.copyfile(src=code_template, dst=dst_path)
            replace_in_file(file_path=dst_path, replacements=replacements)

            workspace_dst_path = get_workspace_dst(path=code_template)
            shutil.copyfile(src=dst_path, dst=workspace_dst_path)

    def reset(self, next_subtask: str | None = None) -> dict[str, str]:
        """Reset the environment and return the initial observation."""

        self.template_code_reset()

        self.done = False

        task_obs = self._get_common_task_observations()
        task_obs[MemKey.TEMPLATES_RELATIVE_PATH] = self.templates_relative_path
        task_obs[MemKey.AVAILABLE_ACTIONS] = [FeatureEngineeringStageNames.CLASSICAL_TAB_FE]
        self.active_stage = FeatureEngineeringStageNames.CLASSICAL_TAB_FE
        return task_obs

    def answer_parser(self, raw_response):
        return raw_response

    def step(self, action: DSAction) -> tuple[dict, float, bool]:
        """Perform an action and return the next observation, reward, and done."""
        if action.stage_name not in self.stages:
            raise RuntimeError(f"{action.stage_name} not in {self.stages}")

        self.done = False
        task_obs = self._get_common_task_observations()
        # Dictionary to map stages to their respective next stages and actions
        stage_transitions = {
            FeatureEngineeringStageNames.START: {
                'next_stage': FeatureEngineeringStageNames.CLASSICAL_TAB_FE,
                'available_actions': [FeatureEngineeringStageNames.CLASSICAL_TAB_FE],
            },
            FeatureEngineeringStageNames.CLASSICAL_TAB_FE: {
                'next_stage': FeatureEngineeringStageNames.MODEL_TRAINING,
                'available_actions': [FeatureEngineeringStageNames.MODEL_TRAINING],
                'fe_iteration': 2
            },
            FeatureEngineeringStageNames.MODEL_TRAINING: {
                'next_stage': FeatureEngineeringStageNames.SELECT_BEST_MODEL,
                'available_actions': [FeatureEngineeringStageNames.SELECT_BEST_MODEL],
                'extra_obs': {
                    MemKey.MODEL_PERF: self.performance
                },
            },
            FeatureEngineeringStageNames.SELECT_BEST_MODEL: {
                'next_stage': FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES,
                'available_actions': [FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES]
            },
            FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES: {
                'next_stage': FeatureEngineeringStageNames.TERMINATE,
                'available_actions': []
            }
        }
        if self.active_stage == FeatureEngineeringStageNames.CLASSICAL_TAB_FE:
            fe_code = action.hyps[DSActionHypKeys.CODE_BLANK]
            fe_code_file_name = os.path.join(self.workspace_path, "trials", f"fe_{self.step_num}.py")
            with open(fe_code_file_name, "w") as f:
                f.write(fe_code)
            self.fe_code += fe_code

        if self.active_stage == FeatureEngineeringStageNames.MODEL_TRAINING:
            task_obs[MemKey.CODE_MODEL_TRAINING] = action.hyps[DSActionHypKeys.CODE_BLANK]
            model_code_file_name = os.path.join(self.workspace_path, "trials", f"model{self.step_num}.py")

            with open(model_code_file_name, "w") as f:
                f.write(task_obs[MemKey.CODE_MODEL_TRAINING])

            perf_file_name = os.path.join(self.workspace_path, "performance.txt")
            with open(perf_file_name) as f:
                performance = f.readlines()
                self.performance.update({self.step_num: performance})
                self.fe_code += f"\n {performance}"
                task_obs[MemKey.CODE_FE] = self.fe_code

            train_dataset_path = os.path.join(self.workspace_path, "data", "train_tab_input_map.csv")
            test_dataset_path = os.path.join(self.workspace_path, "data", "test_tab_input_map.csv")
            target_dataset_path = os.path.join(self.workspace_path, "data", "train_tab_target_map.csv")
            shutil.copy(
                train_dataset_path,
                os.path.join(self.workspace_path, "trials", f"train_tab_input_map_{self.step_num}.csv")
            )
            shutil.copy(
                test_dataset_path,
                os.path.join(self.workspace_path, "trials", f"test_tab_input_map_{self.step_num}.csv")
            )
            shutil.copy(
                target_dataset_path,
                os.path.join(self.workspace_path, "trials", f"train_tab_target_map_{self.step_num}.csv")
            )

            fe_stage = stage_transitions.get(FeatureEngineeringStageNames.CLASSICAL_TAB_FE)
            if fe_stage.get('fe_iteration', 1) == 1:
                self.active_stage = FeatureEngineeringStageNames.SELECT_BEST_MODEL
            elif self.step_num < fe_stage.get('fe_iteration', 1):
                self.step_num += 1
                self.active_stage = FeatureEngineeringStageNames.START  # Reset stage for next iteration of FE
            else:
                stage_transitions[FeatureEngineeringStageNames.MODEL_TRAINING]['extra_obs'].update(
                    {MemKey.MODEL_PERF: json.dumps(self.performance)}
                )

                perf_file_name = os.path.join(self.workspace_path, "trials", f"performance.txt")
                with open(perf_file_name, 'w') as json_file:
                    json.dump(self.performance, json_file, indent=4)

        current_stage = stage_transitions.get(self.active_stage)
        task_obs[MemKey.AVAILABLE_ACTIONS] = current_stage['available_actions']
        self.active_stage = current_stage['next_stage']
        task_obs.update(current_stage.get('extra_obs', {}))
        if current_stage['next_stage'] == FeatureEngineeringStageNames.TERMINATE:
            self.done = True

        reward = 0
        return task_obs, reward, self.done

    def _get_common_task_observations(self) -> dict:
        """Helper function to get common task observations."""
        return {
            MemKey.TASK_DESCRIPTION: self.get_description_from_src(filemap_key=FileMap.TASK_DESCRIPTION),
            MemKey.DATA_DESCRIPTION: self.get_description_from_src(filemap_key=FileMap.DATA_DESCRIPTION),
            MemKey.SUMMARIZED_METRIC_DESCRIPTION: self.get_description_from_src(filemap_key=FileMap.METRIC_DESCRIPTION)
        }

    def get_src_path(self) -> str:
        return os.path.join(self.prepared_setup_dir, self.task_id, self.prepared_version)

    def get_element_src_path(self, filemap_key: FileMap.name) -> str:
        """ Get the path of an element from source folder """
        return os.path.join(self.get_src_path(), filemap_key.value)

    def get_description_from_src(self, filemap_key: FileMap.name) -> str:
        path = self.get_element_src_path(filemap_key=filemap_key)
        with open(path) as f:
            description = "".join(f.readlines())

        return description

    @property
    def templates_root_path(self) -> str:
        return f"{BASE_TEMPLATE_PATH}/feature_engineering/"

    @property
    def templates_relative_path(self) -> str:
        return f"experiments/{self.exp_id}/"

    @property
    def templates_path(self) -> str:
        return os.path.join(self.templates_root_path, self.templates_relative_path)
