#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

__all__ = ['EarlyStopper']


class EarlyStopper(object):
  """Detect an non-increasing or decreasing metric. Exit the training.
  Attributes:
      
  """
  def __init__(self, threshold: float, patience: int) -> None:
    assert isinstance(threshold, float)
    assert isinstance(patience, int)
    self.threshold = threshold
    self.patience = patience
    self.history = {}
    
  def should_stop(self, cost: float) -> bool:
    diff = cost

    return

  def early_break(self) -> None:
      pass
  
  def exit_program(self) -> None:
      pass
  
  def exception_break(self, num_loops: int):
      pass
      