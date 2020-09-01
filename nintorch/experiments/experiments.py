#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google colab functions.
"""
from google.colab import drive


class GoogleDrive(object):
    def __init__(self, name: str) -> None:
        assert isinstance(name, str)
        self.drive_name = name
        
    def mount(self) -> None:
        drive.mount(self.drive_name)
    
    @staticmethod
    def apply_change() -> None:
        drive.flush_and_unmount()


