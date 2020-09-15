#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""constructor.py
"""

class Constructor(object):
    """For reconstruct classes given dict(class: dict_init_params).
    """            
    def __init__(self, classes, init_params) -> None:
        """ 
        """
        self.classes = classes
        self._check_callable()
        
        self.init_params = init_params
        
    @classmethod
    def from_dict(cls):
        """
        """
        return cls()

    def _check_callable(self) -> None:
        assert all([callable(c) for c in self.classes])
    
    def reconstruct(self):
        
        return