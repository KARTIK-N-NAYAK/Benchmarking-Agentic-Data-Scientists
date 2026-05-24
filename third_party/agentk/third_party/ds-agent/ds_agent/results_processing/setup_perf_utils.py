from enum import Enum
from typing import NamedTuple

from third_party.data_preprocessing.env import DataPrepStagesStatusNames


class OutcomeName(str, Enum):
    NOT_REACHED = "NOT REACHED"
    SKIPPED = "SKIPPED"
    SUCCESS = DataPrepStagesStatusNames.PASSED.value
    FAILURE = DataPrepStagesStatusNames.FAILED.value


class OutcomeBase(NamedTuple):
    name: OutcomeName
    color: str
    label: str


# Outcome-specific classes
class SetupSuccessOutcome(OutcomeBase):
    name: OutcomeName = OutcomeName.SUCCESS
    color: str = "#2ecc71"
    label: str = "Passed"


class SetupFailureOutcome(OutcomeBase):
    name: OutcomeName = OutcomeName.FAILURE
    color: str = "#e74c3c"
    label: str = "Failed"


class SetupNotReachedOutcome(OutcomeBase):
    name: OutcomeName = OutcomeName.NOT_REACHED
    color: str = "#9b59b6"
    label: str = "Not Reached"


class SetupSkippedOutcome(OutcomeBase):
    name: OutcomeName = OutcomeName.SKIPPED
    color: str = "#dfe6e9"
    label: str = "Skipped"


class SkipReason(str, Enum):
    NO_TAB_INPUTS = "No Tab. input"
    NO_IMG_INPUTS = "No Image input"
    NO_TXT_INPUTS = "No Text input"
    NO_TAB_TARGETS = "No Tab. target"
    NOT_BIN_CLASSIF = "Not Bin. Classif."


STAGE_TO_RESTRICTION: dict[str, SkipReason | None] = {
    'map_tab_input_train': SkipReason.NO_TAB_INPUTS,
    'map_img_input_train': SkipReason.NO_IMG_INPUTS,
    'map_txt_input_train': SkipReason.NO_TXT_INPUTS,
    'column_types': SkipReason.NO_TAB_INPUTS,
    'map_tab_target_train': SkipReason.NO_TAB_TARGETS,
    'transform_tab_target_train': SkipReason.NO_TAB_TARGETS,
    'positive_class': SkipReason.NOT_BIN_CLASSIF,
    'unit_test_tab_train': SkipReason.NOT_BIN_CLASSIF,
    'unit_test_img_train': SkipReason.NO_IMG_INPUTS,
    'unit_test_txt_train': SkipReason.NO_TXT_INPUTS,
    'unit_test_all_train': None,
    'unit_test_dataloader_train': None,
    'map_tab_input_test': SkipReason.NO_TAB_INPUTS,
    'map_img_input_test': SkipReason.NO_IMG_INPUTS,
    'map_txt_input_test': SkipReason.NO_TXT_INPUTS,
    'unit_test_dataloader_test': None,
    'submission_format': None,
    'submission_format_alt': None,
    'metric': None,
    'final_unit_test': None
}


def stage_name_to_label(stage_name: str) -> str | None:
    """ Convert the name of the stage to its descriptive label """

    modality_map: dict[str, str] = {"tab": "Tab.", "img": "Images", "txt": "Text"}

    predefined_map: dict[str, str] = {
        'unit_test_dataloader_test': 'Unit test for Test DL',
        'submission_format': 'Conversion to Submission Format',
        'submission_format_alt': 'Conversion to Alt. Submission Format',
        'metric': 'Implement metric',
        'column_types': 'Get the Tab. features dtype',
        'positive_class': 'Identify Positive Class (Bin. Classif.)',
        'final_unit_test': 'Final unit test',
    }

    def extract_info(name: str) -> tuple[str, str, str]:
        mod_, in_out_, train_test_ = name.split("_")[-3:]
        mod_text_ = modality_map[mod_]
        in_out_ = in_out_.capitalize() + "s"
        return mod_text_, in_out_, train_test_.capitalize()

    if stage_name in predefined_map:
        return predefined_map[stage_name]

    if stage_name.startswith("map"):
        mod_text, in_out, train_test = extract_info(name=stage_name)
        return f"Create Map for {in_out} {mod_text} {train_test}"

    if stage_name.startswith("transform_tab_target_train"):
        mod_text, in_out, train_test = extract_info(name=stage_name)
        return f"Transform {in_out} {mod_text}"

    if stage_name.startswith("unit_test_dataloader"):
        return f"Unit test for {stage_name.split('_')[-1].capitalize()} DL"

    if stage_name.startswith("unit_test"):
        mod, train_test = stage_name.split("_")[-2:]
        if mod == "all":
            mod_text = "all the "
        else:
            mod_text = "the " + modality_map[mod]
        return f'Unit test for {mod_text} {train_test} maps'

    return None


CORRECT_SETUP_STAGE_ORDER_MAP: dict[str, str] = {stage: stage_name_to_label(stage) for stage in STAGE_TO_RESTRICTION}


def get_setup_success_stages() -> set[str]:
    return {
        DataPrepStagesStatusNames.DONE.value,
        DataPrepStagesStatusNames.PASSED.value,
        DataPrepStagesStatusNames.FORCED.value
    }
