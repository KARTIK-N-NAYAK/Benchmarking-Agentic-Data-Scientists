import argparse
import functools
import runpy
import traceback
from collections import defaultdict
from collections.abc import Collection
from datetime import date
from pathlib import Path

from pydantic import BaseModel
from torchvision import models
from tqdm import tqdm
from transformers import PreTrainedModel

from ds_agent.competition_ids import CompetitionID
from ds_agent.competition_instances import ALL_COMPETITIONS_DICT
from ds_agent.file_map import FileMap

""" Date ResNET 
- Paper release https://arxiv.org/abs/1512.03385: 10 Dec, 2015
- Used in https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview: 1 July, 2016
- Torch version closely reproducing the results (https://github.com/pytorch/vision/commit/1531bf5edd0bc28ac9f4ab88bce0cc422f04c398): 9 Jan, 2017
- Torch updated version (https://github.com/pytorch/vision/commit/11bd2eaa6d6976129836b329b01d1300babddcc9,
    https://github.com/pytorch/vision/issues/3995): 22 March, 2022 
"""
DATE_ResNet = date(2016, 7, 1)

DATE_MOBILENETV2_IMAGENETK_V1 = date(
    2019, 5, 10
)  # https://github.com/d-li14/mobilenetv2.pytorch/commit/18dc590148fb466359487f255458e41153021490 obtaining same accuracy as the torchvision
# https://github.com/pytorch/vision/commits/main/torchvision/models/efficientnet.py
DATE_EfficientNetV1 = date(2021, 8, 26)
DATE_EfficientNetV2 = date(2022, 3, 2)
# https://github.com/pytorch/vision/commits/main/torchvision/models/convnext.py
DATE_ConvNeXt = date(2022, 2, 1)
# https://github.com/pytorch/vision/commits/main/torchvision/models/regnet.py
DATE_RegNet = date(2021, 9, 29)
DATE_RegNetUpdate = date(2021, 10, 5)
# https://github.com/pytorch/vision/commit/6272c412cb6a4f5dbe69b1bc588ac3cfeb77c1cb
DATE_ResNeXt = date(2019, 5, 19)
DATE_V2_REFRESH = date(2022, 3, 22)

RELEASE_DATES_WEIGHTS = {
    models.ResNet50_Weights.IMAGENET1K_V1: DATE_ResNet,
    models.ResNet50_Weights.IMAGENET1K_V2: DATE_V2_REFRESH,
    models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1: DATE_ConvNeXt,
    models.ConvNeXt_Small_Weights.IMAGENET1K_V1: DATE_ConvNeXt,
    models.ConvNeXt_Base_Weights.IMAGENET1K_V1: DATE_ConvNeXt,
    models.ConvNeXt_Large_Weights.IMAGENET1K_V1: DATE_ConvNeXt,
    models.EfficientNet_B0_Weights.IMAGENET1K_V1: DATE_EfficientNetV1,
    models.EfficientNet_B3_Weights.IMAGENET1K_V1: DATE_EfficientNetV1,
    models.EfficientNet_B4_Weights.IMAGENET1K_V1: DATE_EfficientNetV1,
    models.EfficientNet_B7_Weights.IMAGENET1K_V1: DATE_EfficientNetV1,
    models.EfficientNet_V2_L_Weights.IMAGENET1K_V1: DATE_EfficientNetV2,
    models.EfficientNet_V2_M_Weights.IMAGENET1K_V1: DATE_EfficientNetV2,
    models.EfficientNet_V2_S_Weights.IMAGENET1K_V1: DATE_EfficientNetV2,
    models.MobileNet_V2_Weights.IMAGENET1K_V1: DATE_MOBILENETV2_IMAGENETK_V1,
    models.MobileNet_V3_Large_Weights.IMAGENET1K_V1: date(
        2021, 1, 14
    ),  # https://github.com/pytorch/vision/commits/main/torchvision/models/mobilenetv3.py
    models.RegNet_X_8GF_Weights.IMAGENET1K_V1: DATE_RegNet,
    models.RegNet_X_16GF_Weights.IMAGENET1K_V1: DATE_RegNetUpdate,
    models.RegNet_X_32GF_Weights.IMAGENET1K_V1: DATE_RegNetUpdate,
    models.RegNet_X_400MF_Weights.IMAGENET1K_V1: DATE_RegNet,
    models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1: DATE_RegNetUpdate,
    models.RegNet_Y_16GF_Weights.IMAGENET1K_V1: DATE_RegNetUpdate,
    models.RegNet_Y_32GF_Weights.IMAGENET1K_V1: DATE_RegNetUpdate,
    models.RegNet_Y_400MF_Weights.IMAGENET1K_V1: DATE_RegNet,
    models.RegNet_Y_800MF_Weights.IMAGENET1K_V1: DATE_RegNet,
    models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1: DATE_ResNeXt,
    models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1: DATE_ResNeXt,
    models.ResNet101_Weights.IMAGENET1K_V1: DATE_ResNet,
    models.SqueezeNet1_1_Weights.IMAGENET1K_V1: date(
        2017, 2, 11
    ),  # https://github.com/pytorch/vision/commits/main/torchvision/models/squeezenet.py
    models.Swin_T_Weights.IMAGENET1K_V1: date(
        2022, 4, 27
    ),  # https://github.com/pytorch/vision/commits/main/torchvision/models/swin_transformer.py
    models.ViT_B_16_Weights.IMAGENET1K_V1: date(
        2022, 1, 10
    ),  # https://github.com/pytorch/vision/commits/main/torchvision/models/vision_transformer.py
    models.MaxVit_T_Weights.IMAGENET1K_V1: date(
        2022, 9, 23
    ),  # https://github.com/pytorch/vision/commit/6b1646cae7e4a0118bd769cd9921ff2269561047
}

RELEASE_DATES_TRANSFORMERS = {
    "bert-base-uncased": date(2018, 11, 14),  # https://huggingface.co/google-bert/bert-base-uncased/commits/main
    "roberta-large": date(2019, 8, 5),  # https://huggingface.co/FacebookAI/roberta-large/commits/main
    "albert/albert-base-v1": date(2019, 12, 20),  # https://huggingface.co/albert/albert-base-v1/commits/main
    "HooshvareLab/bert-fa-base-uncased": date(
        2020, 5, 26
    ),  # https://huggingface.co/HooshvareLab/bert-fa-base-uncased/commits/main
    "HooshvareLab/bert-fa-zwnj-base": date(
        2021, 2, 18
    ),  # https://huggingface.co/HooshvareLab/bert-fa-zwnj-base/commits/main
    "albert-base-v2": date(2019, 11, 4),  # https://huggingface.co/albert-base-v2/commits/main
    "allenai/scibert_scivocab_uncased": date(
        2020, 3, 18
    ),  # https://huggingface.co/allenai/scibert_scivocab_uncased/commits/main
    "allenai/specter": date(2021, 1, 26),  # https://huggingface.co/allenai/specter/commits/main
    "allenai/longformer-base-4096": date(2020, 5, 18),
    # https://huggingface.co/allenai/longformer-base-4096/commits/main
    "bert-base-multilingual-cased": date(
        2018, 11, 30
    ),  # https://huggingface.co/bert-base-multilingual-cased/commits/main
    "distilbert-base-uncased": date(2019, 8, 28),  # https://huggingface.co/distilbert-base-uncased/commits/main
    "distilroberta-base": date(2019, 10, 17),
    # https://huggingface.co/distilbert/distilrobertdistilbert-base-multilingual-caseda-base/commits/main
    "distilbert-base-multilingual-cased": date(2019, 11, 25),
    "google/electra-base-discriminator": date(
        2020, 3, 24
    ),  # https://huggingface.co/google/electra-base-discriminator/commits/main
    "google/electra-large-discriminator": date(
        2020, 3, 24
    ),  # https://huggingface.co/google/electra-large-discriminator/commits/main
    "microsoft/deberta-base": date(2020, 7, 3),  # https://huggingface.co/microsoft/deberta-base/commits/main
    "microsoft/deberta-v3-base": date(2021, 10, 18),  # https://huggingface.co/microsoft/deberta-v3-base/commits/main
    "microsoft/deberta-v3-large": date(2021, 10, 18),  # https://huggingface.co/microsoft/deberta-v3-large/commits/main
    "roberta-base": date(2019, 8, 3),  # https://huggingface.co/FacebookAI/roberta-base/commits/main
    "sentence-transformers/all-MiniLM-L6-v2": date(
        2021, 8, 30
    ),  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commits/main
    "sentence-transformers/distiluse-base-multilingual-cased": date(
        2021, 6, 22
    ),  # https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/commits/main
    "sentence-transformers/distiluse-base-multilingual-cased-v2": date(
        2021, 6, 22
    ),
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1": date(
        2021, 1, 11
    ),  # https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1/commits/main
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": date(
        2021, 6, 2
    ),
    "xlm-roberta-base": date(2019, 12, 18),  # https://huggingface.co/FacebookAI/xlm-roberta-base/commits/main
    "xlm-roberta-large": date(2019, 12, 18),  # https://huggingface.co/FacebookAI/xlm-roberta-large/commits/main
}

RELEASE_DATES_TABULAR = {
    "catboost": date(2017, 6, 29),  # https://arxiv.org/abs/1706.09516
    "lgbm": date(2017, 5, 7),  # https://app.readthedocs.org/projects/lightgbm/builds/?page=275
    "xgboost": date(2016, 3, 10),  # https://arxiv.org/abs/1603.02754
}


class TooRecentCheck(BaseModel):
    name: str
    release_date: date
    competition_end: date | None
    is_success: bool


class TooRecentException(Exception):
    def __init__(self, message, check_desc: TooRecentCheck):
        super().__init__(message)
        self.check_desc = check_desc
        self.check_desc.is_success = False


class MissingEntry(Exception):
    def __init__(self, message: str, entry_key: str):
        super().__init__(message)
        self.entry_key = entry_key


def verify_check_age(latest_date: date, check_history: list, allow_none=False):
    original_verify = getattr(models.WeightsEnum.verify, "__wrapped__", models.WeightsEnum.verify.__func__)

    @functools.wraps(original_verify)
    def verify(cls: models.WeightsEnum, obj):
        if not allow_none and obj is None:
            raise ValueError("You are using a pretrained architecture but randomly initialising weights=None")
        else:
            release_date = RELEASE_DATES_WEIGHTS.get(obj)
            if release_date is None:
                raise MissingEntry(
                    f"Unrecognized model weights, please register it in ds_agent.code_inspect.model_age: {obj}", obj
                )
            check_desc = TooRecentCheck(
                name=str(obj), release_date=release_date, competition_end=latest_date, is_success=True
            )
            check_history.append(check_desc)
            if release_date > latest_date:
                raise TooRecentException(
                    f"These model weights are too new to be used with this task, "
                    f"try setting `weights` to an older version: {release_date} > {latest_date}",
                    check_desc,
                )

        return original_verify(cls, obj)

    return classmethod(verify)


def from_pretrained_check_age(latest_date: date, check_history: list):
    from_pretrained_original = getattr(
        PreTrainedModel.from_pretrained, "__wrapped__", PreTrainedModel.from_pretrained.__func__
    )

    @functools.wraps(from_pretrained_original)
    def from_pretrained(cls: PreTrainedModel, pretrained_model_name_or_path, *args, **kwargs):
        release_date = RELEASE_DATES_TRANSFORMERS.get(pretrained_model_name_or_path)
        if release_date is None:
            raise MissingEntry(
                f"Unrecognized model name, "
                f"please register it in ds_agent.code_inspect.model_age: {pretrained_model_name_or_path}",
                pretrained_model_name_or_path,
            )
        check_desc = TooRecentCheck(
            name=pretrained_model_name_or_path,
            release_date=release_date,
            competition_end=latest_date,
            is_success=True,
        )
        check_history.append(check_desc)
        if release_date > latest_date:
            raise TooRecentException(
                f"This model is too new to be used with this task, "
                f"try using an older model: {release_date} > {latest_date}",
                check_desc,
            )

        return from_pretrained_original(cls, pretrained_model_name_or_path, *args, **kwargs)

    return classmethod(from_pretrained)


def patch_WeightEnum_verify(latest_date: date) -> list[TooRecentCheck]:
    """Adds age checks to WeightsEnum.verify
    Returns list that is updated with checks that are conducted"""
    check_history = []
    models.WeightsEnum.verify = verify_check_age(latest_date=latest_date, check_history=check_history)
    return check_history


def patch_PreTrainedModel_from_pretrained(latest_date: date) -> list[TooRecentCheck]:
    """Adds age checks to WeightsEnum.verify"""
    check_history = []
    PreTrainedModel.from_pretrained = from_pretrained_check_age(latest_date=latest_date, check_history=check_history)
    return check_history


def test_img_embed_date(p: Path, latest_date: date) -> TooRecentCheck:
    assert p.name == "img_embed.py"
    img_embed = runpy.run_path(str(p))
    check_history = patch_WeightEnum_verify(latest_date=latest_date)
    _ = img_embed["ImageEmbedder"]()
    return check_history[-1]


def test_txt_embed_date(p: Path, latest_date: date) -> TooRecentCheck:
    assert p.name == "txt_embed.py"
    txt_embed = runpy.run_path(str(p))
    check_history = patch_PreTrainedModel_from_pretrained(latest_date=latest_date)
    _ = txt_embed["TextEmbedder"]()
    return check_history[-1]


def check_time_consistent_workspace(
        competition_id: CompetitionID, workspace_path: Path, skip_checked: bool,
        write_json: str | None = FileMap.TOO_RECENT_CHECK.value
) -> dict[..., set]:
    """

    Args:
        competition_id: id of the competition
        workspace_path: path to the workspace containing the model components
        skip_checked: whether to skip running the check if check was already run
        write_json: path where to save the check results

    Returns:
        dictionary mapping error type to messages
    """
    assert competition_id.value in workspace_path.parts

    competition = ALL_COMPETITIONS_DICT[competition_id]
    end_date = competition.leaderboard_end_date.end_date
    if end_date is None:
        end_date = date.max
    else:
        end_date = end_date.date()

    errors = defaultdict(set)
    for p in workspace_path.glob("*_embed.py"):
        check_desc = None
        modality = p.stem.split("_")[0]
        json_path = workspace_path / f"{modality}_{write_json}"
        if json_path.exists() and skip_checked:
            continue

        try:
            if modality == "txt":
                check_desc = test_txt_embed_date(p, latest_date=end_date)
            elif modality == "img":
                check_desc = test_img_embed_date(p, latest_date=end_date)
            else:
                continue
        except MissingEntry as e:
            errors[type(e).__name__].add(e.entry_key)
        except TooRecentException as e:
            check_desc = e.check_desc
            errors[type(e).__name__].add((p, e))
        except Exception as e:  # Including TooRecentException
            errors[type(e).__name__].add(p)
            traceback.print_exc()

        if check_desc is not None and write_json is not None:
            print(f"Writing finished check to {json_path}")
            json_path.write_text(check_desc.model_dump_json(indent=2))

    return dict(errors)


def check_time_consistent_workspaces(
        workspace_paths: Collection[Path | str], skip_checked: bool,
        write_json: str | None = FileMap.TOO_RECENT_CHECK.value
) -> dict[str, set]:
    """

    Args:
        workspace_paths: paths to the workspaces to check
        skip_checked: whether to skip the check of already checked workspaces
        write_json: name of the json to create to write the result

    Returns:
        errors
    """
    all_errors = defaultdict(set)
    for workspace_path in tqdm(map(Path, workspace_paths), total=len(workspace_paths)):
        comp_id = CompetitionID(workspace_path.parents[3].name)
        errors = check_time_consistent_workspace(
            competition_id=comp_id,
            workspace_path=workspace_path,
            skip_checked=skip_checked,
            write_json=write_json
        )

        for k, v in errors.items():
            if isinstance(v, tuple):
                v = v[0]
            all_errors[k].update(v)

    all_errors = dict(all_errors)
    return all_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check release date of the models used in a given competition.")
    parser.add_argument("--workspaces", type=str, nargs="+", required=True, help="Path to workspaces to check")
    parser.add_argument("--skip", type=int, required=True, help="Whether to skip already checked workspaces")
    parser.add_argument("--slugs", type=str, nargs="+", required=True, help="Competition names")

    args = parser.parse_args()

    workspaces = [Path(w) for w in args.workspaces]
    slugs = args.slugs
    skip = args.skip
    for w, slug in zip(workspaces, slugs):
        result = check_time_consistent_workspace(
            competition_id=CompetitionID.get_enum_element(slug),
            workspace_path=w,
            skip_checked=skip,
            write_json=FileMap.TOO_RECENT_CHECK.value
        )
        print(w)
        if not result:
            print("Success!")
        else:
            print(result)
