"""Synthetic gyms for general System One distillation."""

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.debate_judge import generate_debate_episode
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.resource_allocator import generate_allocator_episode
from systemone_lite.synth.sokoban import generate_sokoban_samples
from systemone_lite.synth.ticket_dungeon import generate_ticket_episode

__all__ = [
    "generate_allocator_episode",
    "generate_connect4_samples",
    "generate_debate_episode",
    "generate_game2048_samples",
    "generate_gridworld_samples",
    "generate_sokoban_samples",
    "generate_ticket_episode",
]
