from ds_agent.utils import ListableEnum


class FileMap(str, ListableEnum):
    SETUP_PLAN_JSON = "plan.json"
    TRAIN_TABULAR_INPUT = "train_tab_input_map.csv"
    TRAIN_IMAGE_INPUT = "train_img_input_map.csv"
    TRAIN_TEXT_INPUT = "train_txt_input_map.csv"

    TRAIN_TABULAR_TARGET = "train_tab_target_map.csv"
    TRAIN_IMAGE_TARGET = "train_img_target_map.csv"
    TRAIN_TEXT_TARGET = "train_txt_target_map.csv"

    TEST_TABULAR_INPUT = "test_tab_input_map.csv"
    TEST_IMAGE_INPUT = "test_img_input_map.csv"
    TEST_TEXT_INPUT = "test_txt_input_map.csv"

    METRIC_SCRIPT = "code_metric.py"
    TAB_INV_TRANSFORM = "df_tab_target_inv_transform.csv"
    TXT_INV_TRANSFORM = "df_txt_target_inv_transform.csv"
    IMG_INV_TRANSFORM = "df_img_target_inv_transform.csv"
    TAB_TARGETS_STRUCT = "tab_targets_struct.py"
    COLUMN_TYPES = "column_types.json"
    TARGET_COL_CLASSIFICATION_TRANSFORMS = 'target_columns_transform.json'
    SUBMISSION_NAMES = "submission_names.json"
    SUBMISSION_FORMAT_SCRIPT = "code_submission_format.py"
    SUBMISSION_FORMAT_ALT_SCRIPT = "code_submission_format_alt.py"

    TAB_TARGETS_TRANSFORM = "code_transform_tab_target_train.py"
    TAB_REGRESSION_TARGETS_TRANSFORM = "tab_regression_target_scaler.py"
    IMG_TARGETS_TRANSFORM = "code_transform_img_target_train.py"
    TXT_TARGETS_TRANSFORM = "code_transform_txt_target_train.py"

    DATA_DESCRIPTION = "metadata/data_description.txt"
    TASK_DESCRIPTION = "metadata/task_description.txt"
    METRIC_DESCRIPTION = "metadata/metric_description.txt"
    SUBMISSION_FORMAT = "metadata/submission_format.txt"
    MODALITY_TRANSFORMS_DESCRIPTION = "metadata/modality_transforms_description.txt"

    TABLE_FE_STATUS = "table_fe.json"
    TABLE_MODEL_STATUS = "table_model.json"
    TABLE_EMBED_STATUS = "table_embed.json"
    TABLE_HEAD_STATUS = "table_head.json"
    IMG_EMBED_STATUS = "img_embed.json"
    IMG_HEAD_STATUS = "img_head.json"
    TXT_EMBED_STATUS = "txt_embed.json"
    TXT_HEAD_STATUS = "txt_head.json"
    SUBMISSION_STATE = "submission.json"
    SUBMISSION_SUMMARY = "summary.txt"

    SUBMISSION_FILE = "submission.csv"
    SUBMISSION_ALT_FILE = "submission_alt.csv"
    BLEND_SUBMISSION_FILE = "blend_submission.csv"
    SAMPLE_SUBMISSION_FILE = "sample_submission.csv"
    SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"

    # --- TRAINING
    SOLVE_SCRIPT = "solve.py"
    TRAIN_UTILS = "train_utils.py"
    SOLVE_COMMON_UTILS = "solve_common_utils.py"
    SOLVE_ERROR_LOG = "run_error.log"
    SOLVE_OUTPUT_LOG = "run_output.log"
    VALIDATION_LOSS_FEEDBACK = "val_scores.json"

    # --- LLM COMPONENTS
    TAB_EMBED = "tab_embed.py"
    IMG_EMBED = "img_embed.py"
    TXT_EMBED = "txt_embed.py"

    TOO_RECENT_CHECK = "tooRecentCheck.json"  # check if a model is too recent compared to the competition release date

    TAB_HEAD = "tab_head.py"
    IMG_HEAD = "img_head.py"
    TXT_HEAD = "txt_head.py"

    IMG_INPUT_TRANSFORM = "img_transform.py"
    TAB_FE = "tab_fe.py"
    CLASS_IMBALANCE = "class_imbalance.py"

    # --- BAG
    BAG_SCRIPT = "bag_submissions.py"
    BAG_ERROR_LOG = "bag_error.log"
    BAG_OUTPUT_LOG = "bag_output.log"

    # --- BLEND
    BLEND_SCRIPT = "blend.py"
    BLEND_ERROR_LOG = "blend_error.log"
    BLEND_OUTPUT_LOG = "blend_output.log"
    BLEND_COMMAND_TXT = "blend_command.txt"

    # --- EXECUTION TIME
    REMAINING_TIME_FILE = "remaining_time.txt"
    CHAT_COMPLETION_RETRIAL_TIME = "chat_completion_retrial_time.txt"

    # --- DATASCIENCE ENV UTILS
    RESUME_CHECKPOINT_FILE = "resume_checkpoint.pkl"

    # --- RAMP
    RAMP_METRIC_NAME = "metric.json"  # TODO: use it
    METADATA = "metadata"  # TODO: use it
