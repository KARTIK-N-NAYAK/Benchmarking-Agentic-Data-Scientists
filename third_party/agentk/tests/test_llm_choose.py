from functools import partial

import hydra
from omegaconf import OmegaConf

from agent.commands import ConsiderAction
from agent.commands import DecisionFlow
from agent.commands import Think
from agent.loggers import ManyLoggers
from agent.models.openai_api import OpenAIAPILanguageBackend
from agent.utils import break_word_split


class FakeLogger:
    def __init__(self) -> None:
        pass

    def update_state(self, *args, **kwargs) -> None:
        pass

    def log_metrics(self, *args, **kwargs) -> None:
        pass

    def save_metrics(self) -> None:
        pass


def return_fake_llm(responses) -> OpenAIAPILanguageBackend:
    llm = OpenAIAPILanguageBackend("1", FakeLogger(), 1, "1", "1")
    llm.history = [{"input": "PLACEHOLDER", "output": "PLACEHOLDER", "parsed_response": "PLACEHOLDER"}] * len(responses)
    responses = iter(responses)
    llm.chat_completion = lambda x, y: next(responses)
    return llm


# Test finding the llm response in the list of choices -----------------------------------------------------------------


def test_choose_perfect_match(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    responses = ["command", "dosomething", "some choice"]
    llm = return_fake_llm(responses)
    agent.llm = llm
    options = ["command", "dosomething", "some choice"]

    for i in range(3):
        assert (
                agent.choose_from_options(
                    llm,
                    [],
                    options,
                    parse_func=partial(break_word_split, "break"),
                )
                == options[i]
        )


def test_different_break_words(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    responses = ["Command: command", "Action: action", "Object: object"]
    llm = return_fake_llm(responses)
    agent.llm = llm
    options = ["command", "action", "object"]

    for i, break_word in enumerate(["Command", "Action", "Object"]):
        assert (
                agent.choose_from_options(
                    llm,
                    [],
                    options,
                    parse_func=partial(break_word_split, break_word),
                )
                == options[i]
        )


# Test the string similarity functionality -----------------------------------------------------------------------------


def test_choose_syntax_mismatch(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    responses = ["open, drawer 1", "go to drawer 2!", "goto dresser 1", ".look.", "jump", "wait", "open door 2"]
    llm = return_fake_llm(responses)
    agent.llm = llm
    options = ["open drawer 1", "go to drawer 2", "go to dresser 1", "look", "jump!", "wait...", "open, door 2"]

    for i in range(3):
        assert (
                agent.choose_from_options(
                    llm,
                    [],
                    options,
                    parse_func=partial(break_word_split, "break"),
                )
                == options[i]
        )


def test_choose_spelling_and_word_mismatch(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    responses = ["open drawers 1", "walk towards drawer 2", "choose dreser 1", "look around"]
    llm = return_fake_llm(responses)
    agent.llm = llm
    options = ["open drawer 1", "go to drawer 2", "go to dresser 1", "look"]

    for i in range(3):
        assert (
                agent.choose_from_options(
                    llm=llm,
                    messages=[],
                    options=options,
                    parse_func=partial(break_word_split, "break"),
                )
                == options[i]
        )


def test_choose_wrong(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    responses = ["walk towards drawer 2", "choose dreser 1", "look", "open, drawers 1"]
    llm = return_fake_llm(responses)
    options = ["open drawer 1", "go to drawer 2", "go to dresser 1", "look"]
    agent.llm = llm

    for i in range(4):
        choice = agent.choose_from_options(
            llm=llm, messages=[], options=options, parse_func=partial(break_word_split, "break")
        )
        assert choice != options[i], choice


# Test choose real commands -------------------------------------------------------------------------------------------


def test_choose_real_commands_1(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)

    responses = ["Command: 'think'"]
    agent.llm = return_fake_llm(responses)

    cmds = [Think(), ConsiderAction()]
    flow = DecisionFlow(cmds)

    assert flow.step(agent) is cmds[0]


def test_choose_real_commands_2(cfg_alfworld) -> None:
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)

    responses = ["Command: something incorrect", "Command: select external action"]
    agent.llm = return_fake_llm(responses)

    cmds = [Think(), ConsiderAction()]
    flow = DecisionFlow(cmds)

    assert flow.step(agent) is cmds[1]
