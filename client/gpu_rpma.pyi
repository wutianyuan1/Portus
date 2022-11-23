#
# Automatically generated file, do not edit!
#

"""fast DNN checkpoint with RPMA"""
from __future__ import annotations
import gpu_rpma
import typing

__all__ = [
    "checkpoint",
    "init_checkpoint",
    "restore",
    "wait_checkpoint_done"
]


def checkpoint(use_async: bool = True) -> int:
    """
    do checkpointing
    """
def init_checkpoint(network_name: str, state_dict: typing.Dict[str, at::Tensor], ib_local_addr: str = '192.168.10.101', server_addr: str = '192.168.10.4') -> int:
    """
    init checkpoint
    """
def restore() -> int:
    """
    do restore
    """
def wait_checkpoint_done() -> int:
    """
    wait checkpoint done
    """
