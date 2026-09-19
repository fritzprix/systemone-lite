"""Synthetic gyms for general System One distillation."""

from systemone_lite.synth.debate_judge import generate_debate_episode
from systemone_lite.synth.resource_allocator import generate_allocator_episode
from systemone_lite.synth.ticket_dungeon import generate_ticket_episode

__all__ = [
    "generate_allocator_episode",
    "generate_debate_episode",
    "generate_ticket_episode",
]
