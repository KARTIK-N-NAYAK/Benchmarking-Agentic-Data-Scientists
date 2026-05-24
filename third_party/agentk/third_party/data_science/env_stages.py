from __future__ import annotations

from typing import Callable

import numpy as np

from agent.utils.utils import ListableEnum


class DataScienceStageNames(str, ListableEnum):
    _TOP = "top"

    ADOPT_DNN_FRAMEWORK = "Use deep learning-based solutions."

    ADOPT_RAMP_FRAMEWORK = "Use RAMP solution"

    ADOPT_CLASSICAL_ML_FRAMEWORK = "Use classical ML solutions"

    ADD_SUBMISSION = "Add a submission"

    ADD_DNN_SUBMISSION = "Add a deep neural net based submission"
    CLASS_IMBALANCE = "Handle class imbalance"
    DNN_TAB_EMBEDDING = "Implement tabular embedding"
    TAB_FEATURE_ENGINEERING = "Implement tabular feature engineering"
    TAB_EMBED_PREPROCESSED = "Implement embedding of pre-processed tabular data"
    TRAIN_IMAGE_TRANSFORM = "Implement train image transform"
    TEST_IMAGE_TRANSFORM = "Implement test image transform"
    IMAGE_MODELLING = "Implement image modelling"
    IMAGE_EMBEDDING = "Implement image embedding"
    TEXT_EMBEDDING = "Implement text embedding"
    TABLE_HEAD = "Implement table model head"
    TABLE_REGRESSION_TARGET_TRANSFORM = "Implement transform of the tabular regression target"
    IMAGE_HEAD = "Implement image model head"
    TEXT_HEAD = "Implement text model head"
    ADD_TRAIN_CODE = "Add training code"
    ADD_HYPERPARAMETERS = "Add hyperparameters"
    TRAIN_SUBMISSION = "Train submission"

    # NOT ACTIVE FOR NOW
    ADD_CLASSICAL_TAB_SUBMISSION = "Add a classical ML submission"
    CLASSICAL_TAB_FE = "Add feature engineering"
    CLASSICAL_TAB_MODEL = "Add a predictive model"
    CLASSICAL_TAB_HYP = "Suggest hyperparameters"
    CLASSICAL_TAB_TRAIN = "Train the model"
    CLASSICAL_TAB_SUBMIT = "Submit tab predictions"

    # BAG DNN
    BAG_SUBMISSIONS = "Bag existing submissions"
    GEN_BAG_CODE = "Generate bagging code"

    # BLEND DNN
    BLEND_SUBMISSIONS = "Blend existing submissions"

    # Send DNN
    SEND_SUBMISSION = "Send submission"

    # RAMP stages
    BLEND_RAMP_SUBMISSIONS = "Blend submissions"  # all stages need to have different values (DecisionFlow choices)
    SEND_RAMP_SUBMISSIONS = "Send a submission"  # send a submission to Kaggle


class DependencyStage:
    def __init__(self, stage: DataScienceStage, dependencies: list[Callable[[], bool]]):
        self.stage = stage
        self.dependencies = dependencies

    def is_reachable(self) -> bool:
        """ Whether the stage can be reached (all dependencies have been cleared) """
        return np.all([dependency() for dependency in self.dependencies])

    def __repr__(self) -> str:
        return f"Dependency on {self.stage.name}"


class DataScienceStage:
    substages: list[DependencyStage]
    n_times: int = 0  # number of time this stage has been encountered
    _target_n_times: int = 1  # number of time this stage should be encountered

    def __init__(self, name: DataScienceStageNames):
        self.name = name
        self.active_substage: DataScienceStage | None = None

    def is_fully_over(self) -> bool:
        return self.n_times >= self._target_n_times

    def update(self, stage_name: str) -> None:
        """ Update the current status of the stage """
        if stage_name == self.name.value:
            if len(self.substages) == 0:
                self.n_times += 1
            return

        if self.active_substage is None:
            not_found = True
            for substage in self.substages:
                if substage.stage.name.value == stage_name:
                    self.active_substage = substage.stage
                    substage.stage.reset()
                    not_found = False
                    break
            if not_found:
                raise ValueError(f"No substage {stage_name} for current stage {stage_name}")

        # update active stage
        self.active_substage.update(stage_name=stage_name)
        if self.active_substage.is_fully_over():
            self.active_substage = None

        # If all substages are over or not reachable then the current stage is over
        if np.all([substage.stage.is_fully_over() or not substage.is_reachable() for substage in self.substages]):
            self.n_times += 1

    def get_available_stages(self) -> list[DataScienceStage]:
        """ Get the list of the possible next stage """
        if self.active_substage is not None:  # fetch available actions from active substage
            return self.active_substage.get_available_stages()

        # Otherwise: check dependencies
        available_stages = []
        for dependency_substage in self.substages:
            if dependency_substage.stage.is_fully_over():
                continue
            if dependency_substage.is_reachable():
                available_stages.append(dependency_substage.stage)
        assert len(available_stages) > 0
        return available_stages

    def reset(self) -> None:
        """ Reset this stage and substages """
        self.n_times = 0
        self.active_substage = None
        for dependency_stage in self.substages:
            dependency_stage.stage.reset()


class FeatureEngineering(DataScienceStage):
    substages = []


class TabularModeling(DataScienceStage):
    substages = []


class ImageModelling(DataScienceStage):
    substages = []


class TrainImageTransform(DataScienceStage):
    substages = []


class TestImageTransform(DataScienceStage):
    substages = []


class TextEmbedding(DataScienceStage):
    substages = []


class ClassImbalance(DataScienceStage):
    substages = []


class TabularEmbedding(DataScienceStage):
    def __init__(self, name: DataScienceStageNames) -> None:
        super().__init__(name=name)
        self.feat_engineering = FeatureEngineering(name=DataScienceStageNames.TAB_FEATURE_ENGINEERING)
        self.tabular_modeling = TabularModeling(name=DataScienceStageNames.TAB_EMBED_PREPROCESSED)
        self.substages = [
            DependencyStage(stage=self.feat_engineering, dependencies=[]),
            DependencyStage(stage=self.tabular_modeling, dependencies=[self.feat_engineering.is_fully_over])
        ]


class ImageEmbedding(DataScienceStage):
    def __init__(self, name: DataScienceStageNames) -> None:
        super().__init__(name=name)
        self.train_image_transform = TrainImageTransform(name=DataScienceStageNames.TRAIN_IMAGE_TRANSFORM)
        self.test_image_transform = TestImageTransform(name=DataScienceStageNames.TEST_IMAGE_TRANSFORM)
        self.image_modelling = ImageModelling(name=DataScienceStageNames.IMAGE_MODELLING)
        self.substages = [
            DependencyStage(stage=self.train_image_transform, dependencies=[]),
            DependencyStage(stage=self.test_image_transform, dependencies=[self.train_image_transform.is_fully_over]),
            DependencyStage(stage=self.image_modelling, dependencies=[self.test_image_transform.is_fully_over])
        ]


class FuseImageHead(DataScienceStage):
    substages = []


class FuseTextHead(DataScienceStage):
    substages = []


class FuseTabularHead(DataScienceStage):
    substages = []


class TabularTargetTransform(DataScienceStage):
    substages = []


class AddTrainCode(DataScienceStage):
    substages = []


class AddHyperparameters(DataScienceStage):
    substages = []


class TrainSubmission(DataScienceStage):
    substages = []


class AddDNNSubmission(DataScienceStage):
    """ Solution involving DNN encoder / decoder for each type of inputs / outputs involved """

    def __init__(self, has_table_input: bool, has_img_input: bool, has_txt_input: bool, has_table_target: bool,
                 has_regression_target: bool, has_img_target: bool, has_txt_target: bool,
                 has_classification_target: bool,
                 name: DataScienceStageNames) -> None:
        super().__init__(name=name)
        self.has_table_input = has_table_input
        self.has_img_input = has_img_input
        self.has_txt_input = has_txt_input
        self.has_table_target = has_table_target
        self.has_img_target = has_img_target
        self.has_txt_target = has_txt_target

        self.tabular_embedding = TabularEmbedding(name=DataScienceStageNames.DNN_TAB_EMBEDDING)
        self.image_embedding = ImageEmbedding(name=DataScienceStageNames.IMAGE_EMBEDDING)
        self.text_embedding = TextEmbedding(name=DataScienceStageNames.TEXT_EMBEDDING)
        self.tabular_target_transform = TabularTargetTransform(
            name=DataScienceStageNames.TABLE_REGRESSION_TARGET_TRANSFORM)
        self.class_imbalance = ClassImbalance(name=DataScienceStageNames.CLASS_IMBALANCE)
        self.fuse_tabular_head = FuseTabularHead(name=DataScienceStageNames.TABLE_HEAD)
        self.fuse_text_head = FuseTextHead(name=DataScienceStageNames.TEXT_HEAD)
        self.fuse_image_head = FuseImageHead(name=DataScienceStageNames.IMAGE_HEAD)
        self.add_train_code = AddTrainCode(name=DataScienceStageNames.ADD_TRAIN_CODE)
        self.add_hyperparameters = AddHyperparameters(name=DataScienceStageNames.ADD_HYPERPARAMETERS)
        self.train_submission = TrainSubmission(name=DataScienceStageNames.TRAIN_SUBMISSION)

        self.substages = []

        if has_table_input:
            self.substages.append(DependencyStage(stage=self.tabular_embedding, dependencies=[]))
        if has_img_input:
            self.substages.append(DependencyStage(stage=self.image_embedding, dependencies=[]))
        if has_txt_input:
            self.substages.append(DependencyStage(stage=self.text_embedding, dependencies=[]))
        if has_table_target:
            if has_classification_target:
                self.substages.append(
                    DependencyStage(stage=self.class_imbalance, dependencies=[self.embeddings_are_over])
                )
                fuse_tabular_head_dependence = self.class_imbalance.is_fully_over
            elif has_regression_target:
                self.substages.append(
                    DependencyStage(stage=self.tabular_target_transform, dependencies=[self.embeddings_are_over])
                )
                fuse_tabular_head_dependence = self.tabular_target_transform.is_fully_over
            else:
                fuse_tabular_head_dependence = self.embeddings_are_over
            self.substages.append(
                DependencyStage(stage=self.fuse_tabular_head, dependencies=[fuse_tabular_head_dependence])
            )
        if has_img_target:
            self.substages.append(DependencyStage(stage=self.fuse_image_head, dependencies=[self.embeddings_are_over]))
        if has_txt_target:
            self.substages.append(DependencyStage(stage=self.fuse_text_head, dependencies=[self.embeddings_are_over]))

        self.substages.append(DependencyStage(stage=self.add_train_code, dependencies=[self.fusing_is_over]))

        self.substages.append(
            DependencyStage(
                stage=self.add_hyperparameters,
                dependencies=[self.add_train_code.is_fully_over, self.is_train_submission_not_over]
            )
        )

        self.substages.append(
            DependencyStage(stage=self.train_submission, dependencies=[self.add_train_code.is_fully_over]))

    def is_train_submission_not_over(self):
        return not self.train_submission.is_fully_over()

    def embeddings_are_over(self) -> bool:
        table_over = (not self.has_table_input) or self.tabular_embedding.is_fully_over()
        image_over = (not self.has_img_input) or self.image_embedding.is_fully_over()
        text_over = (not self.has_txt_input) or self.text_embedding.is_fully_over()
        return table_over and image_over and text_over

    def fusing_is_over(self) -> bool:
        table_over = (not self.has_table_target) or self.fuse_tabular_head.is_fully_over()
        image_over = (not self.has_img_target) or self.fuse_image_head.is_fully_over()
        text_over = (not self.has_txt_target) or self.fuse_text_head.is_fully_over()
        return table_over and image_over and text_over


class ClassicalTabFE(DataScienceStage):
    """ Feature engineering in classical ML solution for tabular data """
    substages = []


class ClassicalTabModel(DataScienceStage):
    """ Modelling in classical ML solution for tabular data """
    substages = []


class ClassicalTabHyp(DataScienceStage):
    """ Hyperparametrize the classical ML solution for tabular data """
    substages = []


class ClassicalTabTrain(DataScienceStage):
    """ Train the classical ML solution for tabular data """
    substages = []


class AddTabularOnlySubmission(DataScienceStage):

    def __init__(self, name: DataScienceStageNames):
        super().__init__(name=name)
        self.fe = ClassicalTabFE(name=DataScienceStageNames.CLASSICAL_TAB_FE)
        # model = ClassicalTabModel(name=DataScienceStageNames.CLASSICAL_TAB_MODEL)  # let's only do FE  for now
        # hyp = ClassicalTabHyp(name=DataScienceStageNames.CLASSICAL_TAB_HYP)
        # train = ClassicalTabTrain(name=DataScienceStageNames.CLASSICAL_TAB_TRAIN)

        self.substages = [
            DependencyStage(stage=self.fe, dependencies=[]),
            # DependencyStage(stage=model, dependencies=[fe.is_fully_over]),
            # DependencyStage(stage=hyp, dependencies=[model.is_fully_over]),
            # DependencyStage(stage=train, dependencies=[hyp.is_fully_over]),
        ]


class AddSubmission(DataScienceStage):
    """ The flow to create the submission depends on the type of inputs / outputs involved """

    def __init__(self, has_table_input: bool, has_img_input: bool, has_txt_input: bool,
                 has_table_target: bool, has_regression_target: bool, has_img_target: bool, has_txt_target: bool,
                 has_classification_target: bool, name: DataScienceStageNames) -> None:
        super().__init__(name=name)

        action = AddDNNSubmission(
            has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
            has_table_target=has_table_target, has_regression_target=has_regression_target,
            has_img_target=has_img_target, has_txt_target=has_txt_target,
            has_classification_target=has_classification_target,
            name=DataScienceStageNames.ADD_DNN_SUBMISSION
        )
        self.substages = [DependencyStage(stage=action, dependencies=[])]


# class SelectSubmissions(DataScienceStage):
#     substages = []


# class GenerateBlendingCode(DataScienceStage):
#     substages = []

class Blend(DataScienceStage):
    substages = []


class Bag(DataScienceStage):
    substages = []
    # def __init__(self, name: DataScienceStageNames) -> None:
    #     super().__init__(name=name)
    #     self.select_submissions = SelectSubmissions(name=DataScienceStageNames.SELECT_SUBMISSIONS)
    #     self.generate_blending_code = GenerateBlendingCode(name=DataScienceStageNames.GEN_BLEND_CODE)
    #     self.substages = [
    #         DependencyStage(stage=self.select_submissions, dependencies=[]),
    #         DependencyStage(stage=self.generate_blending_code, dependencies=[self.select_submissions.is_fully_over])
    #     ]


class Submit(DataScienceStage):
    substages = []


class ClassicalTabSubmit(DataScienceStage):
    substages = []


class TopDNNDatascienceStage(DataScienceStage):

    def __init__(self, has_table_input: bool, has_img_input: bool, has_txt_input: bool,
                 has_table_target: bool, has_regression_target: bool, has_img_target: bool,
                 has_txt_target: bool, has_classification_target: bool) -> None:
        super().__init__(name=DataScienceStageNames.ADOPT_DNN_FRAMEWORK)
        self.n_submissions = 0
        self.add_submission = AddSubmission(
            has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
            has_table_target=has_table_target, has_regression_target=has_regression_target,
            has_img_target=has_img_target, has_txt_target=has_txt_target,
            has_classification_target=has_classification_target, name=DataScienceStageNames.ADD_SUBMISSION
        )
        self.submit = Submit(name=DataScienceStageNames.SEND_SUBMISSION)
        self.bag_submissions = Bag(name=DataScienceStageNames.BAG_SUBMISSIONS)
        self.blend_submissions = Blend(name=DataScienceStageNames.BLEND_SUBMISSIONS)
        self.substages = [
            DependencyStage(stage=self.add_submission, dependencies=[]),
            DependencyStage(stage=self.bag_submissions, dependencies=[self.is_multiple_submissions_available]),
            DependencyStage(stage=self.blend_submissions, dependencies=[self.is_multiple_submissions_available]),
            DependencyStage(stage=self.submit, dependencies=[self.has_submissions])
        ]

    def is_multiple_submissions_available(self):
        return self.n_submissions > 1

    def has_submissions(self):
        return self.n_submissions > 0

    def get_available_stages(self) -> list[DataScienceStage]:
        """ Get the list of the possible next stage --- restore top-level stages if needs be"""
        if self.active_substage is None:
            for substage in self.substages:
                if substage.stage.is_fully_over():
                    substage.stage.reset()
        return super().get_available_stages()

    def update(self, stage_name: str) -> None:
        if stage_name == self.add_submission.name.value:
            self.n_submissions += 1
        super().update(stage_name=stage_name)

    def reset(self) -> None:
        self.n_submissions = 0
        super().reset()


class TopClassicalMLRoute(DataScienceStage):
    _target_n_times: int = 10  # number of time this stage should be encountered

    def __init__(self) -> None:
        super().__init__(name=DataScienceStageNames.ADOPT_CLASSICAL_ML_FRAMEWORK)
        self.n_submissions = 0

        self.add_tab_submission = AddTabularOnlySubmission(name=DataScienceStageNames.ADD_CLASSICAL_TAB_SUBMISSION)
        self.submit = ClassicalTabSubmit(name=DataScienceStageNames.CLASSICAL_TAB_SUBMIT)

        self.substages = [
            DependencyStage(stage=self.add_tab_submission, dependencies=[]),
            DependencyStage(stage=self.submit, dependencies=[lambda: self.n_submissions > 10]),
        ]

    def get_available_stages(self) -> list[DataScienceStage]:
        """ Get the list of the possible next stage --- restore top-level stages if needs be"""
        if self.active_substage is None:
            for substage in self.substages:
                if substage.stage.is_fully_over():
                    substage.stage.reset()
        return super().get_available_stages()

    def update(self, stage_name: str) -> None:
        if stage_name == self.add_tab_submission.name.value:
            self.n_submissions += 1
        super().update(stage_name=stage_name)

    def reset(self) -> None:
        self.n_submissions = 0
        super().reset()


class SubmitRamp(DataScienceStage):
    substages = []


class BlendRamp(DataScienceStage):
    substages = []


class TopRAMPDatascienceStage(DataScienceStage):
    """ Datascience flow using actions compatible with the actions of RAMP """

    def __init__(self) -> None:
        super().__init__(name=DataScienceStageNames.ADOPT_RAMP_FRAMEWORK)

        # self.blend_ramp = BlendRamp(name=DataScienceStageNames.BLEND_RAMP_SUBMISSIONS)
        self.submit_ramp = SubmitRamp(name=DataScienceStageNames.SEND_RAMP_SUBMISSIONS)
        self.substages = [
            # DependencyStage(stage=self.blend_ramp, dependencies=[]),
            DependencyStage(stage=self.submit_ramp, dependencies=[]),
        ]

    def get_available_stages(self) -> list[DataScienceStage]:
        """ Get the list of the possible next stage --- restore top-level stages if needs be"""
        if self.active_substage is None:
            # reactivate all options except kit setup
            for substage in self.substages:
                if substage.stage.is_fully_over():
                    substage.stage.reset()
        return super().get_available_stages()


class TopDatascienceStage(DataScienceStage):

    def __init__(self, has_table_input: bool, has_img_input: bool, has_txt_input: bool,
                 has_table_target: bool, has_regression_target: bool, has_img_target: bool,
                 has_txt_target: bool, has_classification_target: bool) -> None:
        super().__init__(name=DataScienceStageNames._TOP)
        self._target_n_times = 1_000_000  # No limit on the number of times we can retry that
        self.has_table_input = has_table_input
        self.has_img_input = has_img_input
        self.has_txt_input = has_txt_input
        self.has_table_target = has_table_target
        self.has_img_target = has_img_target
        self.has_txt_target = has_txt_target

        self.substages = []

        self.ramp_route = None
        self.classical_ml_route = None
        self.dnn_route = None
        self.submit = None

        if self.fully_tabular_task():
            # self.ramp_route = TopRAMPDatascienceStage()
            self.classical_ml_route = TopClassicalMLRoute()

            self.substages.extend([
                # DependencyStage(stage=self.ramp_route, dependencies=[]),
                DependencyStage(stage=self.classical_ml_route, dependencies=[])
            ])
        else:
            self.dnn_route = TopDNNDatascienceStage(
                has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
                has_table_target=has_table_target, has_regression_target=has_regression_target,
                has_img_target=has_img_target, has_txt_target=has_txt_target,
                has_classification_target=has_classification_target
            )
            self.substages.append(DependencyStage(stage=self.dnn_route, dependencies=[]))
            self.submit = self.dnn_route.submit

    def update(self, stage_name: str) -> None:
        if self.submit is None:
            if stage_name == self.classical_ml_route.name.value:
                self.submit = self.classical_ml_route.submit
            elif stage_name == self.ramp_route.name.value:
                self.submit = self.ramp_route.submit_ramp

        super().update(stage_name=stage_name)

    def fully_tabular_task(self) -> bool:
        full_table = self.has_table_input and self.has_table_target
        no_img = not (self.has_img_input or self.has_img_target)
        no_txt = not (self.has_txt_input or self.has_txt_target)
        return full_table and no_txt and no_img
