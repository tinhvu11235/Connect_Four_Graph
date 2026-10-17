"""Tests of ``common.connect4_rules``: gravity, turns, illegal moves, wins in 4 directions, draw."""

import pytest

from common.connect4_rules import ConnectFourEnv


def play(actions):
    """Return a new game after playing ``actions`` (columns 1..7).  Side effects: none."""

    env = ConnectFourEnv()
    for action in actions:
        env.step(action)
    return env


def test_gravity_and_turns():
    """Pieces stack from the bottom and the turn alternates."""

    env = play([4, 4])
    assert env.state[3, 0] == 1 and env.state[3, 1] == -1
    assert env.turn == "red"


def test_full_column_rejected():
    """A full column leaves ``validinputs`` and playing it raises ``ValueError``."""

    env = play([1] * 6)
    assert 1 not in env.validinputs
    with pytest.raises(ValueError):
        env.step(1)


@pytest.mark.parametrize("actions", [
    [1, 1, 2, 2, 3, 3, 4],              # horizontal
    [1, 2, 1, 2, 1, 2, 1],              # vertical
    [1, 2, 2, 3, 3, 4, 3, 4, 4, 7, 4],  # diagonal /
    [7, 6, 6, 5, 5, 4, 5, 4, 4, 1, 4],  # diagonal \
])
def test_red_wins(actions):
    """Four in a row in every direction ends the game for red and blocks further moves."""

    env = play(actions)
    assert env.done and env.reward == 1 and env.turn == "red"
    assert len(env.winning_cells) >= 4
    with pytest.raises(RuntimeError):
        env.step(5)


def test_draw():
    """A full board without four in a row is a draw (reward 0)."""

    order = [1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1,
             3, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4, 3,
             5, 6, 5, 6, 5, 6, 6, 5, 6, 5, 6, 5,
             7, 7, 7, 7, 7, 7]
    env = play(order)
    assert env.done and env.reward == 0
