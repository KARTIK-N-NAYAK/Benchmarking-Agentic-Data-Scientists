from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, validator

from ds_agent.competition_ids import CompetitionID
from ds_agent.utils import ListableEnum


class CompetitionType(str, ListableEnum):
    """Types of competitions"""

    REGRESSION = "Regression"
    MULTI_TARGET_REGRESSION = "Multi-target Regression"
    REGRESSION_AND_BINARY_CLASSIFICATION = "Regression and Binary Classification"

    # NOTE:
    #   - Multiclass Classification is a single-target task with multiple labels
    #   - Multi-target classification is multi-target task with potentially multiple labels but can also be binary
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    MULTI_TARGET_CLASSIFICATION = "Multi-target Classification"
    IMAGE_SEGMENTATION = "Image Segmentation"  # not yet supported

    @staticmethod
    def get_classification_types() -> list[CompetitionType]:
        return [
            CompetitionType.REGRESSION_AND_BINARY_CLASSIFICATION,
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.MULTI_TARGET_CLASSIFICATION,
        ]

    @staticmethod
    def is_classification_only(competition_type) -> bool:
        return competition_type in [
            CompetitionType.BINARY_CLASSIFICATION, CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.MULTI_TARGET_CLASSIFICATION
        ]


class CompetitionTypeAux(BaseModel):
    competition_type: CompetitionType
    circle_colors: list[str]
    short_text: list[str]

    @staticmethod
    def get_competition_types() -> dict[CompetitionType, CompetitionTypeAux]:
        all_comp_types = {
            CompetitionType.REGRESSION: CompetitionTypeAux(
                competition_type=CompetitionType.REGRESSION, circle_colors=["rgb(0, 128, 128)"], short_text=["RG"]
            ),
            CompetitionType.MULTICLASS_CLASSIFICATION: CompetitionTypeAux(
                competition_type=CompetitionType.MULTICLASS_CLASSIFICATION, circle_colors=["rgb(138, 43, 226)"],
                short_text=["MC"]
            ),
            CompetitionType.MULTI_TARGET_CLASSIFICATION: CompetitionTypeAux(
                competition_type=CompetitionType.MULTI_TARGET_CLASSIFICATION, circle_colors=["rgb(178, 34, 34)"],
                short_text=["MTC"]
            ),
            CompetitionType.BINARY_CLASSIFICATION: CompetitionTypeAux(
                competition_type=CompetitionType.BINARY_CLASSIFICATION, circle_colors=["rgb(46, 139, 87)"],
                short_text=["BC"]
            ),
            CompetitionType.MULTI_TARGET_REGRESSION: CompetitionTypeAux(
                competition_type=CompetitionType.MULTI_TARGET_REGRESSION, circle_colors=["rgb(30, 144, 255)"],
                short_text=["MTR"]
            ),
        }

        all_comp_types[CompetitionType.REGRESSION_AND_BINARY_CLASSIFICATION] = CompetitionTypeAux(
            competition_type=CompetitionType.REGRESSION_AND_BINARY_CLASSIFICATION,
            circle_colors=[all_comp_types[comp_type].circle_colors[0] for comp_type in
                           [CompetitionType.REGRESSION, CompetitionType.BINARY_CLASSIFICATION]],
            short_text=[all_comp_types[comp_type].short_text[0] for comp_type in
                        [CompetitionType.REGRESSION, CompetitionType.BINARY_CLASSIFICATION]]
        )

        for k, v in all_comp_types.items():
            assert k == v.competition_type, k
        return all_comp_types


class DataType(str, ListableEnum):
    """Type of data (for inputs / targets)"""

    TAB = "tab"
    IMG = "img"
    TXT = "txt"


class SubmissionMode(str, ListableEnum):
    """Possible Submission modes"""

    FILE_UPLOAD = "file_upload"
    NOTEBOOK_FILE_UPLOAD = "notebook_file_upload"
    CODE_NOTEBOOK = "code_notebook"
    CLOSED = "closed"  # -> we cannot submit anymore


class LeaderboardEndDate(BaseModel):
    is_rolling: bool
    end_date: datetime | None


class Metric(str, ListableEnum):
    """Evaluation metric"""

    MSE = "mse"
    RMSE = "rmse"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    AU_ROC = "au_roc"
    LOG_LOSS = "log_loss"
    MAP_5 = "map_5"
    R_2 = "r_2"
    RMSLE = "rmsle"
    MAE = "mae"
    MCRMSE = "mcrmse"
    MAP_3 = "map_3"
    APPROXIMATE_MEDIAN_SIGNIFICANCE = "approximate_median_significance"
    GINI_COEFF = "gini_coefficient"
    BALANCED_ACCURACY = "balanced_accuracy"
    AVG_PRECISION_K = "average_precision_k"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    NORMALIZED_GINI_COEFFECIENT = "normalized_gini_coeffecient"
    RMS_PERCENTAGE_ERROR = "root_mean_squared_percentage_error"
    QUADRATIC_WEIGHTED_KAPPA = "quadratic_weighted_kappa"
    SMAPE = "smape"
    ROC_AUC = "roc_auc"
    AVERAGE_AGREEMENT = "average_agreement"
    PEARSON_CORRELATION_COEFFECIENT = "pearson_correlation_coeffecient"
    MEAN_F_SCORE = "mean_f_score"
    MACRO_F1_SCORE = "macro_f1_score"
    ADJUSTED_RAND_INDEX = "adjusted_rand_index"
    ACCOUNTING_PENALTY = "accounting_penalty"
    DICE_COEFFECIENT = "dice_coeffecient"
    WEIGHTED_LOGLOSS = "weighted_logloss"
    MAP_12 = "map_12"

    @staticmethod
    def get_clean_names(metric_list: list[Metric]) -> list[str]:
        """
        Return list of cleaned metric names.
        Args:
            metric_list: Metric names list

        Returns:
            List of cleaned metric names
        """
        clean_metric_names = {
            Metric.MSE: "MSE",
            Metric.RMSE: "RMSE",
            Metric.ACCURACY: "ACCURACY",
            Metric.F1_SCORE: "F1 SCORE",
            Metric.AU_ROC: "AUROC",
            Metric.LOG_LOSS: "LOG LOSS",
            Metric.MAP_5: "MAP@5",
            Metric.R_2: "R²",
            Metric.RMSLE: "RMSLE",
            Metric.MAE: "MAE",
            Metric.MCRMSE: "MCRMSE",
            Metric.MAP_3: "MAP@3",
            Metric.APPROXIMATE_MEDIAN_SIGNIFICANCE: "AMS",
            Metric.GINI_COEFF: "Gini Coefficient",
            Metric.BALANCED_ACCURACY: "Balanced Acc.",
            Metric.AVG_PRECISION_K: "AP@K",
            Metric.MEDIAN_ABSOLUTE_ERROR: "Median Abs. Error",
            Metric.NORMALIZED_GINI_COEFFECIENT: "Norm. Gini Coef.",
            Metric.RMS_PERCENTAGE_ERROR: "RMSPE",
            Metric.QUADRATIC_WEIGHTED_KAPPA: "QWK",
            Metric.SMAPE: "SMAPE",
            Metric.ROC_AUC: "ROC AUC",
            Metric.AVERAGE_AGREEMENT: "Average Agreement",
            Metric.PEARSON_CORRELATION_COEFFECIENT: "Pearson Corr. Coef.",
            Metric.MEAN_F_SCORE: "Mean F Score",
            Metric.MACRO_F1_SCORE: "Macro F1 Score",
            Metric.ADJUSTED_RAND_INDEX: "ARI",
            Metric.ACCOUNTING_PENALTY: "Accounting Penalty",
            Metric.DICE_COEFFECIENT: "Dice Coefficient",
            Metric.WEIGHTED_LOGLOSS: "Weighted Log Loss",
            Metric.MAP_12: "MAP@12"
        }

        clean_metric_list = [clean_metric_names[metric] for metric in metric_list]
        return clean_metric_list


class KaggleCompetitionType(str, ListableEnum):
    """ Possible Submission modes """
    GETTING_STARTED = "getting_started"
    PLAYGROUND = "playground"
    FEATURED = "featured"
    COMMUNITY = "community"
    RESEARCH = "research"
    RECRUITMENT = "recruitment"


class KaggleCompetitionTypeAux(BaseModel):
    kaggle_competition_type: KaggleCompetitionType
    circle_colors: list[str]
    short_text: list[str]

    @staticmethod
    def get_kaggle_competition_types() -> dict[KaggleCompetitionType, KaggleCompetitionTypeAux]:
        all_kaggle_comp_types = {
            KaggleCompetitionType.GETTING_STARTED: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.GETTING_STARTED, circle_colors=["rgb(255,140,0)"], short_text=["GS"],
            ),
            KaggleCompetitionType.PLAYGROUND: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.PLAYGROUND, circle_colors=["rgb(128, 0, 128)"], short_text=["P"],
            ),
            KaggleCompetitionType.FEATURED: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.FEATURED, circle_colors=["rgb(0, 0, 128)"], short_text=["F"],
            ),
            KaggleCompetitionType.COMMUNITY: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.COMMUNITY, circle_colors=["rgb(34, 139, 34)"], short_text=["C"],
            ),
            KaggleCompetitionType.RESEARCH: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.RESEARCH, circle_colors=["rgb(0, 139, 139)"], short_text=["RS"],
            ),
            KaggleCompetitionType.RECRUITMENT: KaggleCompetitionTypeAux(
                kaggle_competition_type=KaggleCompetitionType.RECRUITMENT, circle_colors=["rgb(0, 0, 0)"], short_text=["RC"],
            ),
        }
        for k, v in all_kaggle_comp_types.items():
            assert k == v.kaggle_competition_type, k

        return all_kaggle_comp_types


class Competition(BaseModel):
    competition_id: CompetitionID
    input_types: set[DataType]
    target_types: set[DataType]
    competition_type: CompetitionType
    sample_submission_filename: str
    submission_mode: SubmissionMode
    start_date: datetime
    leaderboard_end_date: LeaderboardEndDate
    perc_private: float  # percentatge of the test data used for the private leaderboard (0% -> only public)
    metric: Metric
    award_medals: bool
    kaggle_competition_type: KaggleCompetitionType
    max_selected_submissions: int | None = None  # max number of submissions that can be selected for evaluation
    n_train_points: int | None = None  # number of inputs to train on
    n_test_points: int | None = None  # number of inputs to make predictions on
    comment: str | None = None

    @validator("input_types", "target_types")
    @classmethod
    def must_not_be_empty(cls, data_types: set[DataType]) -> set[DataType]:
        if len(data_types) == 0:
            raise ValueError("must contain a data type")
        return data_types

    def get_str_input_types(self) -> str:
        """Convert input types into a string showing it as a list"""
        return self.convert_data_types_to_str(data_types=self.input_types)

    def get_str_target_types(self) -> str:
        """Convert target types into a string showing it as a list"""
        return self.convert_data_types_to_str(data_types=self.target_types)

    @staticmethod
    def convert_data_types_to_str(data_types: set[DataType]) -> str:
        return '"[' + ",".join(map(lambda data_type: data_type.value, data_types)) + ']"'

    def is_fully_tabular(self) -> bool:
        return self.input_types == {DataType.TAB} and self.target_types == {DataType.TAB}

    def to_dict(self) -> dict[str, ...]:
        return {
            "competition_id": self.competition_id.value,
            "input_types": sorted([t.value for t in self.input_types]),
            "target_types": sorted([t.value for t in self.target_types]),
            "competition_type": self.competition_type.value,
            "sample_submission_filename": self.sample_submission_filename,
            "submission_mode": self.submission_mode.value,
            "comment": self.comment,
        }

    @property
    def has_tab_input(self) -> bool:
        return DataType.TAB in self.input_types

    @property
    def has_txt_input(self) -> bool:
        return DataType.TXT in self.input_types

    @property
    def has_img_input(self) -> bool:
        return DataType.IMG in self.input_types

    @property
    def comp_name(self) -> str:
        return self.competition_id.value

    def __eq__(self, other: Competition) -> bool:
        return other.competition_id == self.competition_id

    def __hash__(self) -> int:
        return hash(self.competition_id)
