from __future__ import annotations

import abc
import dataclasses
import json
import os
import traceback
from enum import auto
from pathlib import Path
from typing import Union, Iterable

from agent.tasks.datascience_task.utils import FileMap
from agent.utils.utils import ListableEnum, check_code_safety, run_python_code, catch_error_wrap
from dataclasses_json import dataclass_json
from ds_agent.competition_ids import CompetitionID
from ds_agent.competition_struct import DataType

from third_party.data_preprocessing.benchmarking.ds_main_post_benchmark import run_ds_pipeline


class ModalityIdentificationError(Exception):
    pass


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepStageName(abc.ABC):

    @abc.abstractmethod
    def to_str(self) -> str:
        pass


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepCreateStageName(DataPrepStageName):
    modality: str
    split: str
    input: bool = False
    target: bool = False
    map: bool = False
    transform: bool = False

    def __post_init__(self) -> None:
        assert self.input != self.target, (f"`self.input` and `self.target` must be different!"
                                           f" Got {self.input}, {self.target} for {self.modality} {self.split}"
                                           f" ismap:{self.map} istransform:{self.transform}")
        assert self.map != self.transform, "`self.map` and `self.transform` must be different!"
        if self.split == 'test':
            if self.target:
                raise ValueError(f"Cannot have a `self.target=True` if `self.split='test'` "
                                 f"because there are no test targets.")
            if self.transform:
                raise ValueError(f"Cannot have a `self.transform=True` if `self.split='test'` "
                                 f"because there are no test targets.")

    @property
    def in_or_out(self) -> str:
        return 'input' if self.input else 'target'

    @property
    def map_or_tf(self) -> str:
        return 'map' if self.map else 'transform'

    def get_spec(self) -> str:
        return self.get_static_spec(map_or_tf=self.map_or_tf, modality=self.modality, in_or_out=self.in_or_out,
                                    split=self.split)

    @staticmethod
    def get_static_spec(map_or_tf: str, modality: str, in_or_out: str, split: str) -> str:
        return f"{map_or_tf}_{modality}_{in_or_out}_{split}"

    def to_str(self) -> str:
        return f"create_{self.get_spec()}"

    def __eq__(self, other: DataPrepCreateStageName) -> bool:
        if not hasattr(other, "to_str"):
            return False
        return self.to_str() == other.to_str()

    def __str__(self) -> str:
        return self.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepTestStageName(DataPrepStageName):
    split: str
    modality: str = None
    map: bool = False
    transform: bool = False
    dataloader: bool = False

    def __post_init__(self) -> None:
        if self.map:
            assert (not self.transform) and (not self.dataloader)
        if self.transform:
            assert (not self.map) and (not self.dataloader)
        if self.dataloader:
            assert (not self.map) and (not self.transform)
        if self.split == 'test':
            if self.transform:
                raise ValueError(f"Cannot have a `self.transform=True` if `self.split='test'` "
                                 f"because there are no test targets.")

    @property
    def type(self) -> str:
        return 'map' if self.map else 'transform' if self.transform else 'dataloader'

    @property
    def _modality(self) -> str:
        return f"_{self.modality}" if self.modality is not None else ""

    def get_spec(self) -> str:
        return f"{self.type}{self._modality}_{self.split}"

    def to_str(self) -> str:
        return f"unit_test_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()

    def __eq__(self, other: DataPrepTestStageName) -> bool:
        return self.to_str() == other.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepMetricStageName(DataPrepStageName):
    @staticmethod
    def get_spec() -> str:
        return "metric"

    def to_str(self) -> str:
        return f"select_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepColumnTypesStageName(DataPrepStageName):
    @staticmethod
    def get_spec() -> str:
        return "column_types"

    def to_str(self) -> str:
        return f"create_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepPositiveClassStageName(DataPrepStageName):
    @staticmethod
    def get_spec() -> str:
        return "positive_class"

    def to_str(self) -> str:
        return f"create_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepSubmissionFormatStageName(DataPrepStageName):
    @staticmethod
    def get_spec() -> str:
        return "submission_format"

    def to_str(self) -> str:
        return f"create_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()


@dataclass_json
@dataclasses.dataclass(frozen=True)
class DataPrepSubmissionFormatAltStageName(DataPrepStageName):
    @staticmethod
    def get_spec() -> str:
        return "submission_format_alt"

    def to_str(self) -> str:
        return f"create_{self.get_spec()}"

    def __str__(self) -> str:
        return self.to_str()


class DataPrepStagesStatusNames(ListableEnum):
    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return name

    TODO = auto()
    DONE = auto()
    FAILED = auto()
    FORCED = auto()
    PASSED = auto()


@dataclass_json
@dataclasses.dataclass
class DataPrepStageStatus:
    done: bool = False
    status_str: str = DataPrepStagesStatusNames.TODO.value

    def set_done(self) -> None:
        self.done = True
        self.status_str = DataPrepStagesStatusNames.DONE.value

    def set_failed(self) -> None:
        self.status_str = DataPrepStagesStatusNames.FAILED.value
        self.done = False

    def set_forced(self) -> None:
        self.status_str = DataPrepStagesStatusNames.FORCED.value
        self.done = True

    def reset(self) -> None:
        self.status_str = DataPrepStagesStatusNames.TODO.value
        self.done = False

    def __str__(self) -> str:
        return self.status_str


def run_unit_test(
        unit_test: str,
        unit_test_filename: str,
        workspace_path: str,
        templates_relative_path: str,
        path_to_python: str
) -> tuple[float, str | None, str | None, str]:
    """
    Reads unit test from where it is initially written,
    Saves its code in the current workspace and give it the name `unit_test_filename`,
    Runs the unit test code and returns:
        - the reward: 0 if failed and 1 if passed
        - the unit test output if any
        - the unit test error if any
        - the unit_test_filename
    """
    # Run Unit Test
    with open(os.path.join(templates_relative_path, unit_test), "r") as f:
        unit_test_code = f.read()

    check_code_safety(unit_test_code)

    if ".py" not in unit_test_filename:
        unit_test_filename += ".py"
    unit_test_path = os.path.join(workspace_path, unit_test_filename)
    unit_test_output_path = os.path.join(workspace_path, "_unit_test_output.txt")
    unit_test_error_path = os.path.join(workspace_path, "_unit_test_error.txt")
    unit_test_warnings_path = os.path.join(workspace_path, "_unit_test_warnings.txt")
    aux_unit_test_error_path = os.path.join(workspace_path, "_aux_unit_test_error.txt")

    # save code to python file
    wrapped_unit_test_code = catch_error_wrap(
        code=unit_test_code, code_error_path=unit_test_error_path, code_warnings_path=unit_test_warnings_path
    )
    with open(unit_test_path, "w") as f:
        f.writelines(wrapped_unit_test_code)

    # Run code
    unit_test_output, unit_test_warnings, unit_test_error = run_python_code(
        workspace_path=workspace_path,
        path_to_python=path_to_python,
        code_path=unit_test_path,
        code_output_path=unit_test_output_path,
        code_warnings_path=unit_test_warnings_path,
        code_error_path=unit_test_error_path,
        aux_code_error_path=aux_unit_test_error_path,
    )

    if unit_test_error == "":
        if unit_test_warnings != "":
            print(unit_test_warnings)
        return 1, unit_test_output, None, unit_test_filename

    else:
        print(f"Error in unit test:\n\n{unit_test_error}", flush=True)
        return 0, None, unit_test_error, unit_test_filename


@dataclass_json
@dataclasses.dataclass
class DataPrepStage:
    name: Union[
        DataPrepCreateStageName,
        DataPrepTestStageName,
        DataPrepMetricStageName,
        DataPrepSubmissionFormatStageName,
        DataPrepSubmissionFormatAltStageName,
        DataPrepColumnTypesStageName
    ]
    status: DataPrepStageStatus
    unit_tests: list[str]  # list of runnable unit test files
    workspace_path: str
    templates_relative_path: str
    path_to_python: str
    code: str | None = None
    code_summary: str | None = None
    code_output: str | None = None
    unit_tests_info: dict[str, dict[str, ...]] | None = None  # filled after self.get_reward()
    retro_stages: list[DataPrepStage] = None  # stages that can be changed retroactively

    def __post_init__(self) -> None:
        assert not self.status.done, (f"DataPrepStage with name {self.name} cannot be initialized with "
                                      f"DataPrepStageStatus already done!")

    @property
    def is_done(self) -> bool:
        return self.status.done

    @property
    def is_failed(self) -> bool:
        return self.status.status_str == DataPrepStagesStatusNames.FAILED.value

    @property
    def is_forced(self) -> bool:
        return self.status.status_str == DataPrepStagesStatusNames.FORCED.value

    def __str__(self) -> str:
        if self.unit_tests_info:
            unit_tests_info_str = "\n. ".join([
                f'{ut}: {self.unit_tests_info[ut]["status"]}' for ut in self.unit_tests_info
            ])
        else:
            unit_tests_info_str = "Not attempted yet"
        return (rf"""{self.name}:   
            - status [{self.status}]
            - summary: {self.code_summary}
            - unit tests:
                {unit_tests_info_str}""")

    def get_reward(self) -> tuple[float, dict[str, ...]]:
        """
        Runs the unit tests associated with the Stage
        Returns:
            - the total reward which is how many tests passed
            - a dict containing the information about each individual unit test.
        """
        total_reward, info = 0, {}
        assert self.unit_tests is not None and isinstance(self.unit_tests, Iterable) and len(self.unit_tests) > 0
        for unit_test in self.unit_tests:
            info[unit_test] = {}
            reward, output, error, unit_test_filename = run_unit_test(
                unit_test=unit_test,
                unit_test_filename=f"unit_test_{self.name.get_spec()}.py",
                templates_relative_path=self.templates_relative_path,
                workspace_path=self.workspace_path,
                path_to_python=self.path_to_python,
            )
            total_reward += reward
            info[unit_test]["ran"] = True
            info[unit_test]["reward"] = reward
            info[unit_test]["output"] = output
            info[unit_test]["error"] = error
            if reward == 1:
                info[unit_test]["passed"] = True
                info[unit_test]["status"] = DataPrepStagesStatusNames.PASSED.value
            else:
                info[unit_test]["passed"] = False
                info[unit_test]["status"] = DataPrepStagesStatusNames.FAILED.value

        # set stage status after knowing which unit tests passed
        if len(self.unit_tests) == total_reward:
            self.status.set_done()
        else:
            self.status.set_failed()
            # for group unit test stages, if failed, set associated stages to failed a posteriori,
            # so they can be attempted again in the _available_stages of the Plan
            if self.retro_stages is not None:
                for stage in self.retro_stages:
                    stage.status.set_failed()

        self.unit_tests_info = info

        return total_reward, info

    def get_fake_successful_reward(self) -> tuple[float, dict[str, ...]]:
        """
        Used in the case where the stage does not require to run a unit test,
        e.g. if the stage is "select_metric" and the metric is in RAMP, then we don't
        generate code and hence have no need to run any unit test.

        We then simply return what would be returned in the case of a successful unit test.
        """
        total_reward, info = 1., {}
        if self.unit_tests is not None and isinstance(self.unit_tests, Iterable) and len(self.unit_tests) > 0:
            for unit_test in self.unit_tests:
                info[unit_test] = {}
                reward = 1.
                output = "Unit test passed."
                error = None
                total_reward += reward
                info[unit_test]["ran"] = True
                info[unit_test]["reward"] = reward
                info[unit_test]["output"] = output
                info[unit_test]["error"] = error
                info[unit_test]["passed"] = True
                info[unit_test]["status"] = DataPrepStagesStatusNames.PASSED.value
        else:
            info['placeholder_unit_test'] = {}
            reward = 1.
            output = "Unit test passed."
            error = None
            total_reward += reward
            info['placeholder_unit_test']["ran"] = True
            info['placeholder_unit_test']["reward"] = reward
            info['placeholder_unit_test']["output"] = output
            info['placeholder_unit_test']["error"] = error
            info['placeholder_unit_test']["passed"] = True
            info['placeholder_unit_test']["status"] = DataPrepStagesStatusNames.PASSED.value

        self.status.set_forced()
        self.unit_tests_info = info

        return total_reward, info

    def set_code_and_summary_and_output(self, code: str, code_summary: str, code_output: str):
        self.code = code
        self.code_summary = code_summary
        self.code_output = code_output

    def reset(self) -> None:
        self.status.reset()
        self.code = None
        self.code_summary = None
        self.code_output = None
        self.unit_tests_info = None


@dataclass_json
@dataclasses.dataclass
class DataPrepPlan:
    # stages to create
    map_tab_input_train: DataPrepStage = None
    map_img_input_train: DataPrepStage = None
    map_txt_input_train: DataPrepStage = None
    map_tab_target_train: DataPrepStage = None
    map_img_target_train: DataPrepStage = None
    map_txt_target_train: DataPrepStage = None
    transform_tab_target_train: DataPrepStage = None
    transform_img_target_train: DataPrepStage = None
    transform_txt_target_train: DataPrepStage = None
    map_tab_input_test: DataPrepStage = None
    map_img_input_test: DataPrepStage = None
    map_txt_input_test: DataPrepStage = None
    submission_format: DataPrepStage = None
    submission_format_alt: DataPrepStage = None
    metric: DataPrepStage = None
    column_types: DataPrepStage = None
    positive_class: DataPrepStage = None

    # stages to test
    unit_test_tab_train: DataPrepStage = None  # unit test on train tabular input and targets
    unit_test_img_train: DataPrepStage = None  # unit test on train image input and targets
    unit_test_txt_train: DataPrepStage = None  # unit test on train text input and targets
    unit_test_all_train: DataPrepStage = None  # unit test on train input and targets of all modalities
    unit_test_dataloader_train: DataPrepStage = None  # unit test for the train dataloader
    unit_test_dataloader_test: DataPrepStage = None  # unit test for the test dataloader

    _available_stage_names: list[DataPrepCreateStageName | DataPrepTestStageName] = None
    _available_stages: list[DataPrepStage] = None

    _train_input_map_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _train_target_map_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _train_target_tf_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _train_maps_unit_tests: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _test_input_map_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _train_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)
    _test_stages: list[DataPrepStage] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        assert ((self.map_tab_input_train and self.map_tab_input_test) or
                (not self.map_tab_input_train and not self.map_tab_input_test)), \
            "The plan needs to have both train and test input tabular maps!"
        assert ((self.map_img_input_train and self.map_img_input_test) or
                (not self.map_img_input_train and not self.map_img_input_test)), \
            "The plan needs to have both train and test input image maps!"
        assert ((self.map_txt_input_train and self.map_txt_input_test) or
                (not self.map_txt_input_train and not self.map_txt_input_test)), \
            "The plan needs to have both train and test input text maps!"

        self.stages = []
        for stage in [
            self.map_tab_input_train, self.map_img_input_train, self.map_txt_input_train,
            self.map_tab_target_train, self.map_img_target_train, self.map_txt_target_train,
            self.transform_tab_target_train, self.transform_img_target_train, self.transform_txt_target_train,
            self.unit_test_tab_train, self.unit_test_img_train, self.unit_test_txt_train,
            self.unit_test_all_train, self.unit_test_dataloader_train,
            self.map_tab_input_test, self.map_img_input_test, self.map_txt_input_test,
            self.unit_test_dataloader_test,
            self.submission_format, self.submission_format_alt,
            self.metric,
            self.column_types, self.positive_class
        ]:
            if stage is not None:
                self.stages.append(stage)

        # at the beginning, only choice is to attempt the first stage (pretty radical I know)
        self._available_stage_names = [self.stages[0].name]

        # groups of stages
        for stage in [self.map_tab_input_train, self.map_img_input_train, self.map_txt_input_train]:
            if stage is not None:
                self._train_input_map_stages.append(stage)
        for stage in [self.map_tab_target_train, self.map_img_target_train, self.map_txt_target_train]:
            if stage is not None:
                self._train_target_map_stages.append(stage)
        for stage in [self.transform_tab_target_train, self.transform_img_target_train,
                      self.transform_txt_target_train]:
            if stage is not None:
                self._train_target_tf_stages.append(stage)
        for stage in [self.map_tab_input_test, self.map_img_input_test, self.map_txt_input_test]:
            if stage is not None:
                self._test_input_map_stages.append(stage)
        for stage in [self.unit_test_tab_train, self.unit_test_img_train, self.unit_test_txt_train,
                      self.unit_test_all_train]:
            if stage is not None:
                self._train_maps_unit_tests.append(stage)
        for stage in [self.map_tab_input_train, self.map_tab_target_train,
                      self.map_img_input_train, self.map_img_target_train,
                      self.map_txt_input_train, self.map_txt_target_train,
                      self.transform_tab_target_train,
                      self.transform_img_target_train,
                      self.transform_txt_target_train,
                      self.unit_test_tab_train, self.unit_test_img_train, self.unit_test_txt_train,
                      self.unit_test_all_train, self.unit_test_dataloader_train]:
            if stage is not None:
                self._train_stages.append(stage)
        for stage in [self.map_tab_input_test, self.map_img_input_test,
                      self.map_txt_input_test, self.unit_test_dataloader_test]:
            if stage is not None:
                self._test_stages.append(stage)

        # check that required (stage-group-)unit tests are defined
        if self.has_tab_input or self.has_tab_target:
            assert self.unit_test_tab_train is not None, \
                f"Plan has training tabular inputs or targets, it needs to have a train tabular unit test"

        if self.has_img_input or self.has_img_target:
            assert self.unit_test_img_train is not None, \
                f"Plan has training image inputs or targets, it needs to have a train image unit test"

        if self.has_txt_input or self.has_txt_target:
            assert self.unit_test_txt_train is not None, \
                f"Plan has training text inputs or targets, it needs to have a text tabular unit test"

        assert self.unit_test_all_train is not None, "Please define unit test for all train modalities"
        assert self.unit_test_dataloader_train is not None, "Please define unit test for train dataloader"
        assert self.unit_test_dataloader_test is not None, "Please define unit test for test dataloader"

    @property
    def has_tab_input(self) -> bool:
        # assume if it has a training tabular input map it has to also have a test tabular input map
        return self.map_tab_input_train is not None

    @property
    def has_img_input(self) -> bool:
        # assume if it has a training image input map it has to also have a test image input map
        return self.map_img_input_train is not None

    @property
    def has_txt_input(self) -> bool:
        # assume if it has a training text input map it has to also have a test text input map
        return self.map_txt_input_train is not None

    @property
    def has_tab_target(self) -> bool:
        return self.map_tab_target_train is not None

    @property
    def has_img_target(self) -> bool:
        return self.map_img_target_train is not None

    @property
    def has_txt_target(self) -> bool:
        return self.map_txt_target_train is not None

    @property
    def has_tab_target_transform(self) -> bool:
        return self.transform_tab_target_train is not None

    @property
    def has_img_target_transform(self) -> bool:
        return self.transform_img_target_train is not None

    @property
    def has_txt_target_transform(self) -> bool:
        return self.transform_txt_target_train is not None

    @property
    def is_full_tabular(self) -> bool:
        """ Whether the task only has tabular inputs and targets """
        only_tab_input = self.has_tab_input and (not self.has_img_input) and (not self.has_txt_input)
        only_tab_target = self.has_tab_target and (not self.has_img_target) and (not self.has_txt_target)
        return only_tab_input and only_tab_target

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    @property
    def _train_input_maps_are_done(self) -> bool:
        for stage in self._train_input_map_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _train_target_maps_are_done(self) -> bool:
        for stage in self._train_target_map_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _train_target_tfs_are_done(self) -> bool:
        for stage in self._train_target_tf_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _test_input_maps_are_done(self) -> bool:
        for stage in self._test_input_map_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _time_to_do_column_types(self) -> bool:
        return self.is_full_tabular and self._train_input_maps_are_done and not self.column_types.is_done

    @property
    def _time_to_do_positive_class(self) -> bool:
        return (
                self.is_full_tabular
                and self._train_input_maps_are_done
                and self._train_target_maps_are_done
                and self._train_target_tfs_are_done
                and not self.positive_class.is_done
        )

    @property
    def _time_to_do_target_maps(self) -> bool:
        return self._train_input_maps_are_done and not self._train_target_maps_are_done

    @property
    def _time_to_do_transforms(self) -> bool:
        return (
                self._train_input_maps_are_done
                and self._train_target_maps_are_done
                and not self._train_target_tfs_are_done
        )

    @property
    def _train_stages_are_done(self) -> bool:
        for stage in self._train_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _test_stages_are_done(self) -> bool:
        for stage in self._test_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _test_input_map_stages_are_done(self) -> bool:
        for stage in self._test_input_map_stages:
            if not stage.is_done:
                return False
        return True

    @property
    def _time_to_do_train_maps_unit_tests(self) -> bool:
        ready_stages = self._train_input_maps_are_done and self._train_target_maps_are_done
        unit_tests_remaining = [
            self.unit_test_tab_train, self.unit_test_img_train, self.unit_test_txt_train, self.unit_test_all_train
        ]
        # filter out None and Done
        unit_tests_remaining = [stage for stage in unit_tests_remaining if stage is not None]
        unit_tests_remaining = [stage for stage in unit_tests_remaining if not stage.is_done]
        return ready_stages and len(unit_tests_remaining) > 0

    @property
    def _time_to_do_train_dataloader_unit_test(self) -> bool:
        ready_stages = (
                self._train_input_maps_are_done and
                self._train_target_maps_are_done and
                self._train_target_tfs_are_done and
                not self._test_input_maps_are_done
        )
        unit_tests_remaining = [self.unit_test_dataloader_train]
        unit_tests_remaining = [stage for stage in unit_tests_remaining if not stage.is_done]
        return ready_stages and len(unit_tests_remaining) > 0

    @property
    def _time_to_do_test_dataloader_unit_test(self) -> bool:
        ready_stages = self._train_stages_are_done and self._test_input_maps_are_done
        unit_tests_remaining = [self.unit_test_dataloader_test]
        unit_tests_remaining = [stage for stage in unit_tests_remaining if not stage.is_done]
        return ready_stages and len(unit_tests_remaining) > 0

    @property
    def _time_to_do_submission_format(self) -> bool:
        return self._train_stages_are_done and self._test_stages_are_done and not self.submission_format.is_done

    @property
    def _time_to_do_submission_format_alt(self) -> bool:
        return (
                self._train_stages_are_done
                and self._test_stages_are_done
                and self.submission_format.is_done
                and not self.submission_format_alt.is_done
        )

    @property
    def _time_to_do_metric(self) -> bool:
        return (
                self._train_stages_are_done
                and self._test_stages_are_done
                and self.submission_format.is_done
                and self.submission_format_alt.is_done
                and not self.metric.is_done
        )

    def _update_available_stages(self) -> None:
        """Checks status of all stages and updates available ones."""
        done_stages_all = []
        not_done_stages_all = []
        failed_stages_all = []
        todo_stages_all = []
        for stage in self.stages:
            if stage.is_done:
                done_stages_all.append(stage)
            elif stage.is_failed:
                failed_stages_all.append(stage)
                not_done_stages_all.append(stage)
            else:
                todo_stages_all.append(stage)
                not_done_stages_all.append(stage)

        print('-' * 50)
        if os.getenv('PRINT_PLAN', False):
            print('All Stages in DataPrepPlan')
            print('-' * 50)
            for s in self.stages:
                if s is not None:
                    print(s.name.to_str(), s.status)
            print('-' * 50)

        # Stages are done in this order:
        # train input maps > column types (optional) > train target maps > train target transforms >
        # train maps unit tests > train dataloader unit test > test input maps > test dataloader unit test >
        # submission format > metric > DONE
        if not self._train_input_maps_are_done:
            todo_stages = [stage for stage in todo_stages_all if stage in self._train_input_map_stages]
            failed_stages = [stage for stage in failed_stages_all if stage in self._train_input_map_stages]
        elif self._time_to_do_column_types:
            todo_stages = [stage for stage in todo_stages_all if stage == self.column_types]
            failed_stages = [stage for stage in failed_stages_all if stage == self.column_types]
        elif self._time_to_do_target_maps:
            todo_stages = [stage for stage in todo_stages_all if stage in self._train_target_map_stages]
            failed_stages = [stage for stage in failed_stages_all if stage in self._train_target_map_stages]
        elif self._time_to_do_transforms:
            todo_stages = [stage for stage in todo_stages_all if stage in self._train_target_tf_stages]
            failed_stages = [stage for stage in failed_stages_all if stage in self._train_target_tf_stages]
        elif self._time_to_do_positive_class:
            todo_stages = [stage for stage in todo_stages_all if stage == self.positive_class]
            failed_stages = [stage for stage in failed_stages_all if stage == self.positive_class]
        elif self._time_to_do_train_maps_unit_tests:
            todo_stages = [[stage for stage in not_done_stages_all if stage in self._train_maps_unit_tests][0]]
            failed_stages = []
            # failed_stages = [stage for stage in failed_stages_all if stage in self._train_maps_unit_tests]
        elif self._time_to_do_train_dataloader_unit_test:
            todo_stages = [stage for stage in todo_stages_all if stage == self.unit_test_dataloader_train]
            failed_stages = [stage for stage in failed_stages_all if stage == self.unit_test_dataloader_train]
        elif self._train_stages_are_done and not self._test_input_map_stages_are_done:
            todo_stages = [stage for stage in todo_stages_all if stage in self._test_input_map_stages]
            failed_stages = [stage for stage in failed_stages_all if stage in self._test_input_map_stages]
        elif self._time_to_do_test_dataloader_unit_test:
            todo_stages = [stage for stage in todo_stages_all if stage == self.unit_test_dataloader_test]
            failed_stages = [stage for stage in failed_stages_all if stage == self.unit_test_dataloader_test]
        elif self._time_to_do_submission_format:
            todo_stages = [stage for stage in todo_stages_all if stage == self.submission_format]
            failed_stages = [stage for stage in failed_stages_all if stage == self.submission_format]
        elif self._time_to_do_submission_format_alt:
            todo_stages = [stage for stage in todo_stages_all if stage == self.submission_format_alt]
            failed_stages = [stage for stage in failed_stages_all if stage == self.submission_format_alt]
        elif self._time_to_do_metric:
            todo_stages = [stage for stage in todo_stages_all if stage == self.metric]
            failed_stages = [stage for stage in failed_stages_all if stage == self.metric]
        else:
            todo_stages = todo_stages_all
            failed_stages = failed_stages_all

        if os.getenv('PRINT_PLAN', False):
            __next_available_stages = failed_stages + todo_stages
            print('-' * 50)
            print('Next available stages')
            print('-' * 50)
            for s in __next_available_stages:
                if s is not None:
                    print(s.name.to_str(), s.status)
            print('-' * 50)

        self._available_stages = failed_stages + todo_stages
        self._available_stage_names = [stage.name for stage in self._available_stages]

    def get_available_stage_names(self) -> list[DataPrepCreateStageName | DataPrepTestStageName]:
        """ Given the status of all stages, return the list of available stage names. """
        return self._available_stage_names

    def get_available_stages(self) -> list[DataPrepStage]:
        """ Given the status of all stages, return the list of available stages. """
        return self._available_stages

    def get_input_modalities(self) -> set[str]:
        input_modalities = set()
        if self.has_tab_input:
            input_modalities.add("tab")
        if self.has_img_input:
            input_modalities.add("img")
        if self.has_txt_input:
            input_modalities.add("txt")
        return input_modalities

    def get_target_modalities(self) -> set[str]:
        target_modalities = set()
        if self.has_tab_target:
            target_modalities.add("tab")
        if self.has_img_target:
            target_modalities.add("img")
        if self.has_txt_target:
            target_modalities.add("txt")
        return target_modalities

    @property
    def is_done(self) -> bool:
        available_stages = self.get_available_stages()
        if len(available_stages) == 0:
            return True
        else:
            return False

    def get_stages_code(self) -> dict[str, dict[str, str]]:
        """ Given the status of all stages, return the codes and summaries. """
        stages_code_dict = {}
        for stage in self.stages:
            stages_code_dict[stage.name.to_str()] = {}
            if stage.code is not None:
                stages_code_dict[stage.name.to_str()]["code"] = stage.code
            if stage.code_summary is not None:
                stages_code_dict[stage.name.to_str()]["code_summary"] = stage.code_summary
            if stage.code_output is not None:
                stages_code_dict[stage.name.to_str()]["code_output"] = stage.code_output
        return stages_code_dict

    def __str__(self) -> str:
        """Presents the Plan and each step's status in string format"""
        plan_str = "* Train Input maps:"
        if self.map_tab_input_train:
            plan_str += f"\n\t- {self.map_tab_input_train}"
        if self.column_types:
            plan_str += f"\n\t- {self.column_types}"
        if self.map_img_input_train:
            plan_str += f"\n\t- {self.map_img_input_train}"
        if self.map_txt_input_train:
            plan_str += f"\n\t- {self.map_txt_input_train}"

        plan_str += "\n* Train Target maps:"
        if self.map_tab_target_train:
            plan_str += f"\n\t- {self.map_tab_target_train}"
        if self.map_img_target_train:
            plan_str += f"\n\t- {self.map_img_target_train}"
        if self.map_txt_target_train:
            plan_str += f"\n\t- {self.map_txt_target_train}"

        if self.unit_test_tab_train:
            plan_str += f"\n* Unit Test for Train Inputs and Targets of Tabular modality:"
            plan_str += f"\n\t- {self.unit_test_tab_train}"
        if self.unit_test_img_train:
            plan_str += f"\n* Unit Test for Train Inputs and Targets of Image modality:"
            plan_str += f"\n\t- {self.unit_test_img_train}"
        if self.unit_test_txt_train:
            plan_str += f"\n* Unit Test for Train Inputs and Targets of Text modality:"
            plan_str += f"\n\t- {self.unit_test_txt_train}"
        if self.unit_test_all_train:
            plan_str += f"\n* Unit Test for Train Inputs and Targets of All modalities:"
            plan_str += f"\n\t- {self.unit_test_all_train}"

        plan_str += "\n* Train Target transforms:"
        if self.transform_tab_target_train:
            plan_str += f"\n\t- {self.transform_tab_target_train}"
        if self.transform_img_target_train:
            plan_str += f"\n\t- {self.transform_img_target_train}"
        if self.transform_txt_target_train:
            plan_str += f"\n\t- {self.transform_txt_target_train}"

        if self.positive_class:
            plan_str += f"\n\t- {self.positive_class}"

        plan_str += f"\n* Unit Test for Train DataLoader:"
        plan_str += f"\n\t- {self.unit_test_dataloader_train}"

        plan_str += "\n* Test Input maps:"
        if self.map_tab_input_test:
            plan_str += f"\n\t- {self.map_tab_input_test}"
        if self.map_img_input_test:
            plan_str += f"\n\t- {self.map_img_input_test}"
        if self.map_txt_input_test:
            plan_str += f"\n\t- {self.map_txt_input_test}"

        plan_str += f"\n* Unit Test for Test DataLoader:"
        plan_str += f"\n\t- {self.unit_test_dataloader_test}"

        plan_str += "\n* Submission format Function:"
        plan_str += f"\n\t- {self.submission_format}"
        plan_str += "\n* Submission format Alternative Function:"
        plan_str += f"\n\t- {self.submission_format_alt}"

        plan_str += "\n* Metric Function:"
        plan_str += f"\n\t- {self.metric}"

        return plan_str

    def to_str(self) -> str:
        return self.__str__()

    @staticmethod
    def create_plan_from_dict(
            stage_name_dict: dict[str, bool],
            workspace_path: str,
            templates_relative_path: str,
            path_to_python: str,
            unit_tests: dict[str, list[str]] = None,
    ) -> DataPrepPlan:
        """
        Creates and return a DataPrepPlan dataclass from a dictionary of stage names and their presence in the plan.
        """

        stage_dict: dict[str, DataPrepStage] = {}
        # status_plan = {}
        # if os.path.exists(os.path.join(workspace_path, 'plan_status.json')):
        #     with open(os.path.join(workspace_path, 'plan_status.json'), 'r') as json_data:
        #         status_plan = json.load(json_data)
        #     status_plan = {k: eval(v) for k,v in status_plan.items() }
        #     stage_name_dict.update(status_plan)
        if unit_tests is None:
            unit_tests = {}

        if stage_name_dict["tabular_inputs_needed"]:
            train_stage_name = DataPrepCreateStageName(modality="tab", split="train", input=True, map=True)
            map_tab_input_train = DataPrepStage(
                name=train_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(train_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            test_stage_name = DataPrepCreateStageName(modality="tab", split="test", input=True, map=True)
            map_tab_input_test = DataPrepStage(
                name=test_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(test_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_tab_input_train"] = map_tab_input_train
            stage_dict["map_tab_input_test"] = map_tab_input_test

        if stage_name_dict["image_inputs_needed"]:
            train_stage_name = DataPrepCreateStageName(modality="img", split="train", input=True, map=True)
            map_img_input_train = DataPrepStage(
                name=train_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(train_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            test_stage_name = DataPrepCreateStageName(modality="img", split="test", input=True, map=True)
            map_img_input_test = DataPrepStage(
                name=test_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(test_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_img_input_train"] = map_img_input_train
            stage_dict["map_img_input_test"] = map_img_input_test

        if stage_name_dict["text_inputs_needed"]:
            train_stage_name = DataPrepCreateStageName(modality="txt", split="train", input=True, map=True)
            map_txt_input_train = DataPrepStage(
                name=train_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(train_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            test_stage_name = DataPrepCreateStageName(modality="txt", split="test", input=True, map=True)
            map_txt_input_test = DataPrepStage(
                name=test_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(test_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_txt_input_train"] = map_txt_input_train
            stage_dict["map_txt_input_test"] = map_txt_input_test

        if stage_name_dict["tabular_targets_needed"]:
            stage_name = DataPrepCreateStageName(modality="tab", split="train", target=True, map=True)
            map_tab_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_tab_target_train"] = map_tab_target_train

        if stage_name_dict["image_targets_needed"]:
            stage_name = DataPrepCreateStageName(modality="img", split="train", target=True, map=True)
            map_img_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_img_target_train"] = map_img_target_train

        if stage_name_dict["text_targets_needed"]:
            stage_name = DataPrepCreateStageName(modality="txt", split="train", target=True, map=True)
            map_txt_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["map_txt_target_train"] = map_txt_target_train

        if stage_name_dict["tabular_targets_transform_needed"]:
            stage_name = DataPrepCreateStageName(modality="tab", split="train", target=True, transform=True)
            transform_tab_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["transform_tab_target_train"] = transform_tab_target_train

        if stage_name_dict["image_targets_transform_needed"]:
            stage_name = DataPrepCreateStageName(modality="img", split="train", target=True, transform=True)
            transform_img_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["transform_img_target_train"] = transform_img_target_train

        if stage_name_dict["text_targets_transform_needed"]:
            stage_name = DataPrepCreateStageName(modality="txt", split="train", target=True, transform=True)
            transform_txt_target_train = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
            )
            stage_dict["transform_txt_target_train"] = transform_txt_target_train

        # add stage-groups unit tests
        if stage_name_dict["tabular_inputs_needed"] or stage_name_dict["tabular_targets_needed"]:
            stage_name = DataPrepTestStageName(modality="tab", split="train", map=True)
            retro_stages = []
            for name in ["map_tab_input_train", "map_tab_target_train", "column_types"]:
                if name in stage_dict:
                    retro_stages.append(stage_dict[name])
            unit_test_train_tab_maps = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
                retro_stages=retro_stages
            )
            stage_dict["unit_test_tab_train"] = unit_test_train_tab_maps

        if stage_name_dict["image_inputs_needed"] or stage_name_dict["image_targets_needed"]:
            stage_name = DataPrepTestStageName(modality="img", split="train", map=True)
            retro_stages = []
            for name in ["map_img_input_train", "map_img_target_train"]:
                if name in stage_dict:
                    retro_stages.append(stage_dict[name])
            unit_test_train_img_maps = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
                retro_stages=retro_stages
            )
            stage_dict["unit_test_img_train"] = unit_test_train_img_maps

        if stage_name_dict["text_inputs_needed"] or stage_name_dict["text_targets_needed"]:
            stage_name = DataPrepTestStageName(modality="txt", split="train", map=True)
            retro_stages = []
            for name in ["map_txt_input_train", "map_txt_target_train"]:
                if name in stage_dict:
                    retro_stages.append(stage_dict[name])
            unit_test_train_txt_maps = DataPrepStage(
                name=stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python,
                retro_stages=retro_stages
            )
            stage_dict["unit_test_txt_train"] = unit_test_train_txt_maps

        unit_test_map_all_train_name = DataPrepTestStageName(modality="all", split="train", map=True)
        retro_stages = []
        for name in ["map_tab_input_train", "map_img_input_train", "map_txt_input_train",
                     "map_tab_target_train", "map_img_target_train", "map_txt_target_train",
                     "column_types"]:
            if name in stage_dict:
                retro_stages.append(stage_dict[name])
        unit_test_map_all_train = DataPrepStage(
            name=unit_test_map_all_train_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(unit_test_map_all_train_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
            retro_stages=retro_stages
        )
        stage_dict["unit_test_all_train"] = unit_test_map_all_train

        unit_test_dataloader_train_name = DataPrepTestStageName(split="train", dataloader=True)
        retro_stages = []
        for name in ["map_tab_input_train", "map_img_input_train", "map_txt_input_train",
                     "map_tab_target_train", "map_img_target_train", "map_txt_target_train",
                     "transform_tab_target_train", "transform_img_target_train", "transform_txt_target_train",
                     "column_types", "positive_class"]:
            if name in stage_dict:
                retro_stages.append(stage_dict[name])
        unit_test_dataloader_train = DataPrepStage(
            name=unit_test_dataloader_train_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(unit_test_dataloader_train_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
            retro_stages=retro_stages
        )
        stage_dict["unit_test_dataloader_train"] = unit_test_dataloader_train

        unit_test_dataloader_test_name = DataPrepTestStageName(split="test", dataloader=True)
        retro_stages = []
        for name in ["map_tab_input_test", "map_img_input_test", "map_txt_input_test"]:
            if name in stage_dict:
                retro_stages.append(stage_dict[name])
        unit_test_dataloader_test = DataPrepStage(
            name=unit_test_dataloader_test_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(unit_test_dataloader_test_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
            retro_stages=retro_stages
        )
        stage_dict["unit_test_dataloader_test"] = unit_test_dataloader_test

        submission_format_stage_name = DataPrepSubmissionFormatStageName()
        submission_format = DataPrepStage(
            name=submission_format_stage_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(submission_format_stage_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
        )
        stage_dict["submission_format"] = submission_format

        submission_format_alt_stage_name = DataPrepSubmissionFormatAltStageName()
        submission_format_alt = DataPrepStage(
            name=submission_format_alt_stage_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(submission_format_alt_stage_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
        )
        stage_dict["submission_format_alt"] = submission_format_alt

        metric_stage_name = DataPrepMetricStageName()
        metric = DataPrepStage(
            name=metric_stage_name,
            status=DataPrepStageStatus(done=False),
            unit_tests=unit_tests.get(metric_stage_name.get_spec()),
            workspace_path=workspace_path,
            templates_relative_path=templates_relative_path,
            path_to_python=path_to_python,
        )
        stage_dict["metric"] = metric

        plan = DataPrepPlan(**stage_dict)

        # if the plan is tabular-only,
        #   create an additional stage to create the column types
        #   create an additional stage to query the LLM for positive class name (for RAMP)
        #   re-initialize plan
        if plan.is_full_tabular:
            # add column types stage
            column_types_stage_name = DataPrepColumnTypesStageName()
            column_types_stage = DataPrepStage(
                name=column_types_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(column_types_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python
            )
            stage_dict["column_types"] = column_types_stage
            # add positive class stage
            positive_class_stage_name = DataPrepPositiveClassStageName()
            positive_class_stage = DataPrepStage(
                name=positive_class_stage_name,
                status=DataPrepStageStatus(done=False),
                unit_tests=unit_tests.get(positive_class_stage_name.get_spec()),
                workspace_path=workspace_path,
                templates_relative_path=templates_relative_path,
                path_to_python=path_to_python
            )
            stage_dict["positive_class"] = positive_class_stage

            plan = DataPrepPlan(**stage_dict)

        return plan


@dataclass_json
@dataclasses.dataclass
class DataPrepObs:
    plan: DataPrepPlan
    reward: float = None
    info: dict[str, dict[str, ...]] | None = None


@dataclass_json
@dataclasses.dataclass
class DataPrepInfo(DataPrepObs):
    last_seen_stage: DataPrepStage | None = None


class DataPrepStageParam(str, ListableEnum):
    UNIT_TEST_TEMPLATES = "unit_test_templates"
    UNIT_TEST_FILES = "unit_test_files"
    CODE_FILES = "code_files"
    CODE = "code"
    CODE_OUTPUT = "code_output"
    CODE_SUMMARY = "code_summary"
    RAMP_METRIC_SELECTED = "ramp_metric_selected"
    RAMP_METRIC_CODE = "ramp_code_metric"


@dataclass_json
@dataclasses.dataclass
class DataPrepAction:
    stage_name: DataPrepCreateStageName | DataPrepTestStageName
    params: dict[DataPrepStageParam, ...]


class DataPrepEnv:

    def __init__(
            self,
            task_id: str,
            workspace_path: str,
            path_to_python: str,
            templates_relative_path: str,
            plan: DataPrepPlan = None,
            use_final_unit_test: bool = False,
            stage_max_retries: int = 5,
    ):
        self.task_id = task_id
        self.workspace_path = workspace_path
        self.path_to_python = path_to_python
        self.templates_relative_path = templates_relative_path
        self.use_final_unit_test = use_final_unit_test
        self.obs = None
        self.info = None
        self.plan = plan
        self._available_actions = None
        self.status_plan: dict[str, bool] = {}
        # for submission format
        self._submission_format_counter = 0
        self._submission_format_alt_counter = 0
        self._max_retries = stage_max_retries
        self._chat_completion_retrial_runtime = 0  # Time spent retrying to query the LLM

        if os.path.exists(os.path.join(self.workspace_path, 'plan_status.json')):
            with open(os.path.join(self.workspace_path, 'plan_status.json'), 'r') as json_data:
                self.status_plan = json.load(json_data)

    def __str__(self) -> str:
        if self.plan is not None:
            return self.plan.__str__()
        else:
            return "Not initialized DataPrepEnv (did you forget to initialize me? ;) )"

    def reset(self) -> None:
        # Create workspace directory
        os.makedirs(self.workspace_path, exist_ok=True)

        self._chat_completion_retrial_runtime = 0

        if self.plan is not None:
            self.plan.reset()

            if not self.status_plan:

                assert self.plan.has_tab_input or self.plan.has_img_input or self.plan.has_txt_input
                if not self.plan.has_tab_target and not self.plan.has_img_target and not self.plan.has_txt_target:
                    raise ValueError(f"No target - check in\n\t- {self.workspace_path}")

            self._available_actions = self.init_get_available_actions()

    def init_get_available_actions(self) -> list[DataPrepCreateStageName | DataPrepTestStageName]:
        if self.plan.has_tab_input:
            stage_name = DataPrepCreateStageName(modality="tab", split="train", input=True, map=True)
        elif self.plan.has_img_input:
            stage_name = DataPrepCreateStageName(modality="img", split="train", input=True, map=True)
        elif self.plan.has_txt_input:
            stage_name = DataPrepCreateStageName(modality="txt", split="train", input=True, map=True)
        else:
            raise ValueError("Should have some inputs")
        available_actions = [stage_name]
        return available_actions

    def get_available_actions(self) -> list[str]:
        self._available_actions = self.plan.get_available_stage_names()
        return [action.to_str() for action in self._available_actions]

    @property
    def is_done(self) -> bool:
        return self.plan.is_done if self.plan is not None else False

    @staticmethod
    def get_path_to_chat_completion_retrial_time(workspace_path: str) -> str:
        return os.path.join(workspace_path, FileMap.CHAT_COMPLETION_RETRIAL_TIME.value)

    @property
    def path_to_chat_completion_retrial_time(self) -> str:
        return self.get_path_to_chat_completion_retrial_time(workspace_path=self.workspace_path)

    @property
    def input_modalities(self) -> set[DataType]:
        input_modalities = set()
        if self.plan.has_tab_input:
            input_modalities.add(DataType.TAB)
        if self.plan.has_img_input:
            input_modalities.add(DataType.IMG)
        if self.plan.has_txt_input:
            input_modalities.add(DataType.TXT)

        return input_modalities

    @property
    def output_modalities(self) -> set[DataType]:
        output_modalities = set()
        if self.plan.has_tab_target:
            output_modalities.add(DataType.TAB)
        if self.plan.has_img_target:
            output_modalities.add(DataType.IMG)
        if self.plan.has_txt_target:
            output_modalities.add(DataType.TXT)

        return output_modalities

    def step(self, action: DataPrepAction, retrial_chat_completion_time: float) \
            -> tuple[DataPrepObs, float, bool, DataPrepInfo]:
        self._chat_completion_retrial_runtime += retrial_chat_completion_time
        with open(self.path_to_chat_completion_retrial_time, "w") as f:
            f.write(str(self._chat_completion_retrial_runtime))
        if action.stage_name == DataPrepCreateStageName(modality="tab", split="train", input=True, map=True):
            assert self.plan.has_tab_input, (f"Trying to create the training tabular input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_tab_input_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_tab_input_train.get_reward()

        elif action.stage_name == DataPrepColumnTypesStageName():
            assert self.plan.is_full_tabular, (f"Trying to create the column types but plan is not fully tabular!"
                                               f"\n{self.plan}")
            self.plan.column_types.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.column_types.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="img", split="train", input=True, map=True):
            assert self.plan.has_img_input, (f"Trying to create the training image input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_img_input_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_img_input_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="txt", split="train", input=True, map=True):
            assert self.plan.has_txt_input, (f"Trying to create the training text input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_txt_input_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_txt_input_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="tab", split="train", target=True, map=True):
            assert self.plan.has_tab_target, (f"Trying to create the training tabular target map "
                                              f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_tab_target_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_tab_target_train.get_reward()
            # create some table...

        elif action.stage_name == DataPrepCreateStageName(modality="img", split="train", target=True, map=True):
            assert self.plan.has_img_target, (f"Trying to create the training image target map "
                                              f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_img_target_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_img_target_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="txt", split="train", target=True, map=True):
            assert self.plan.has_txt_target, (f"Trying to create the training text target map "
                                              f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_txt_target_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_txt_target_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="tab", split="train", target=True, transform=True):
            assert self.plan.has_tab_target_transform, (f"Trying to create the training tabular target transforms "
                                                        f"but there is no such stage in the plan!\n{self.plan}")
            # If tab target classification no code is generated
            reward, info = self.plan.transform_tab_target_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="img", split="train", target=True, transform=True):
            assert self.plan.has_img_target_transform, (f"Trying to create the training image target transforms "
                                                        f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.transform_img_target_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.transform_img_target_train.get_reward()

        elif action.stage_name == DataPrepCreateStageName(modality="txt", split="train", target=True, transform=True):
            assert self.plan.has_txt_target_transform, (f"Trying to create the training text target transforms "
                                                        f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.transform_txt_target_train.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.transform_txt_target_train.get_reward()

        elif action.stage_name == DataPrepPositiveClassStageName():
            assert self.plan.is_full_tabular, (f"Trying to create the positive class(es) but plan is not fully tabular!"
                                               f"\n{self.plan}")
            # no code generated for positive class, simply the json file saved in metadata
            reward, info = self.plan.positive_class.get_fake_successful_reward()

        elif action.stage_name.to_str().lower() == "create_map_tab_input_test":
            assert self.plan.has_tab_input, (f"Trying to create the test tabular input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_tab_input_test.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_tab_input_test.get_reward()

        elif action.stage_name.to_str().lower() == "create_map_img_input_test":
            assert self.plan.has_img_input, (f"Trying to create the test image input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_img_input_test.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_img_input_test.get_reward()

        elif action.stage_name.to_str().lower() == "create_map_txt_input_test":
            assert self.plan.has_txt_input, (f"Trying to create the test text input map "
                                             f"but there is no such stage in the plan!\n{self.plan}")
            self.plan.map_txt_input_test.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            reward, info = self.plan.map_txt_input_test.get_reward()

        elif action.stage_name.to_str().lower() == "unit_test_map_tab_train":
            assert self.plan.unit_test_tab_train is not None, (
                f"Trying to run the unit test for train tabular input-targets"
                f" but there is no such stage in the plan!\n{self.plan}")
            reward, info = self.plan.unit_test_tab_train.get_reward()
            self.save_plan("unit_test_map_tab_train", reward)

        elif action.stage_name.to_str().lower() == "unit_test_map_img_train":
            assert self.plan.unit_test_img_train is not None, (
                f"Trying to run the unit test for train image input-targets"
                f" but there is no such stage in the plan!\n{self.plan}")
            reward, info = self.plan.unit_test_img_train.get_reward()

        elif action.stage_name.to_str().lower() == "unit_test_map_txt_train":
            assert self.plan.unit_test_txt_train is not None, (
                f"Trying to run the unit test for train text input-targets"
                f" but there is no such stage in the plan!\n{self.plan}")
            reward, info = self.plan.unit_test_txt_train.get_reward()

        elif action.stage_name.to_str().lower() == "unit_test_map_all_train":
            reward, info = self.plan.unit_test_all_train.get_reward()

        elif action.stage_name.to_str().lower() == "unit_test_dataloader_train":
            reward, info = self.plan.unit_test_dataloader_train.get_reward()

        elif action.stage_name.to_str().lower() == "unit_test_dataloader_test":
            reward, info = self.plan.unit_test_dataloader_test.get_reward()

        elif action.stage_name.to_str().lower() == "select_metric":
            if action.params.get(DataPrepStageParam.RAMP_METRIC_SELECTED, False):
                reward, info = self.plan.metric.get_fake_successful_reward()
            else:
                self.plan.metric.set_code_and_summary_and_output(
                    code=action.params[DataPrepStageParam.CODE],
                    code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                    code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
                )
                reward, info = self.plan.metric.get_reward()

        elif action.stage_name.to_str().lower() == "create_submission_format":
            self.plan.submission_format.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            if self._submission_format_counter >= self._max_retries:
                reward, info = self.plan.submission_format.get_fake_successful_reward()
            else:
                reward, info = self.plan.submission_format.get_reward()
                self._submission_format_counter += 1

        elif action.stage_name.to_str().lower() == "create_submission_format_alt":
            self.plan.submission_format_alt.set_code_and_summary_and_output(
                code=action.params[DataPrepStageParam.CODE],
                code_summary=action.params[DataPrepStageParam.CODE_SUMMARY],
                code_output=action.params[DataPrepStageParam.CODE_OUTPUT],
            )
            if self._submission_format_alt_counter >= self._max_retries:
                reward, info = self.plan.submission_format_alt.get_fake_successful_reward()
            else:
                reward, info = self.plan.submission_format_alt.get_reward()
                self._submission_format_counter += 1

        else:
            raise ValueError(f"Unknown action {action}")

        # Update each stage status of the plan and set available stages
        self.plan._update_available_stages()

        # Only return info about the last unit test for memory or the first unsuccessful one
        if self.is_done and self.use_final_unit_test:
            final_info = self.run_final_unit_test()
            info["final"] = final_info

        last_seen_stage = None
        for stage in self.plan.stages:
            if action.stage_name == stage.name:
                last_seen_stage = stage

        self.info = DataPrepInfo(
            plan=self.plan,
            reward=reward,
            info=info,
            last_seen_stage=last_seen_stage
        )
        self.obs = DataPrepObs(
            plan=self.plan,
            reward=reward,
            info=info,
        )

        return self.obs, reward, self.is_done, self.info

    def run_final_unit_test(self) -> dict[str, ...]:
        output_dir = self.workspace_path
        final_info = {"ran": True}

        class DSFailure(Exception):
            def __init__(self, workspace_dir: Path):
                self.trace_str = ""

                run_error_log = workspace_dir / FileMap.SOLVE_ERROR_LOG.value
                error_log = workspace_dir / "error.txt"
                if run_error_log.exists():
                    self.trace_str = f"{FileMap.SOLVE_ERROR_LOG.value}: " + run_error_log.read_text()
                if error_log.exists():
                    self.trace_str += "error.txt: " + error_log.read_text()

                if self.trace_str == "":
                    self.trace_str = f"DS pipeline failed but {error_log} and {run_error_log} do not exist"

                super().__init__(self.trace_str)

        try:
            if self.plan.is_full_tabular:
                # Run ramp test
                from agent.tasks.datascience_task.ramp_utils import prepare_for_ramp_setup
                prepare_for_ramp_setup(
                    info_path=self.workspace_path,
                    data_path=self.workspace_path,
                    challenge_name=self.task_id,
                    output_path=Path(self.workspace_path, "final_unit_test"),
                    post_setup=True
                )
                import ramphy.ramp_setup as rs

                rs.scripts.setup.setup(
                    ramp_kit="final_unit_test",
                    setup_root=output_dir,
                    kit_root=output_dir,
                    version="test",
                    number=0
                )
                os.chmod(output_dir, 0o777)
                # TODO: should save the output somewhere
            else:
                # output_dir should usually be benchmark_dir/experiment_dir/task_id/seed_i/
                workspace_dir = Path(output_dir)
                seed = workspace_dir.name.split("_")[-1]
                experiment_dir = workspace_dir.absolute().parent.parent

                if isinstance(self.task_id, CompetitionID):
                    task_dict = {CompetitionID.get_enum_element(value=self.task_id): [seed]}
                else:
                    task_dict = {self.task_id: [seed]}

                success_dict = run_ds_pipeline(
                    task_dict=task_dict,
                    output_dir=experiment_dir.parent,
                    experiment_name=experiment_dir.name,
                    python_path=self.path_to_python,
                    prepared_setup_dir=Path(self.workspace_path).parent.parent,
                    workspace_path=Path(self.workspace_path),
                    input_modalities=self.input_modalities,
                    output_modalities=self.output_modalities
                )

                success = success_dict[self.task_id][seed]

                if not success:
                    raise DSFailure(workspace_dir)

            final_info["error"] = None
            final_info["status"] = DataPrepStagesStatusNames.PASSED.value
            final_info["passed"] = True
            final_info["reward"] = 1
        except Exception as e:
            final_info["error"] = e.trace_str if isinstance(e, DSFailure) else traceback.format_exc()
            final_info["status"] = DataPrepStagesStatusNames.FAILED.value
            final_info["passed"] = False
            final_info["reward"] = 0
        return final_info

    def save_plan(self, action_stage_name: str, reward: float):
        if reward:
            if action_stage_name == "unit_test_map_tab_train":
                self.status_plan["tabular_inputs_needed"] = False
                self.status_plan["tabular_targets_needed"] = False

            elif action_stage_name == "unit_test_map_img_train":
                self.status_plan["image_inputs_needed"] = False
                self.status_plan["image_targets_needed"] = False

            elif action_stage_name == "unit_test_map_txt_train":
                self.status_plan["text_inputs_needed"] = False
                self.status_plan["text_targets_needed"] = False

            elif action_stage_name == "unit_test_map_all_train":
                self.status_plan["tabular_targets_transform_needed"] = False
                self.status_plan["text_targets_transform_needed"] = False
                self.status_plan["image_targets_transform_needed"] = False

            with open(os.path.join(self.workspace_path, 'plan_status.json'), 'w') as f:
                json.dump(self.status_plan, f)
