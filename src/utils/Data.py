#!/usr/bin/env python
# coding: utf-8

class Data:
    def __init__(self):
        self.CPL = {'X': [], 'Y': []}
        self.PL = {'X': [], 'Y': []}
        self.thickness = None
        self.stretching = None
        self.angle = None
        self.temperature = None
        self.mix_ration = []
        self.ra = None
        self.cie = None
        self.x = None
        self.y = None
        self.z = None


    def __str__(self):
        return (
            f"Data(\n"
            f"  CPL: {self.CPL}\n"
            f"  PL: {self.PL}\n"
            f"  Thickness: {self.thickness}\n"
            f"  Stretching: {self.stretching}\n"
            f"  Angle: {self.angle}\n"
            f"  Annealing Temperature: {self.temperature}\n"
            f"  Mix ration: {self.mix_ration}\n"
            f"  RA: {self.ra}\n"
            f"  CIE: {self.cie}\n"
            f"  X: {self.x}\n"
            f"  Y: {self.y}\n"
            f"  Z: {self.z}\n"
            f")"
        )
    
    def is_fluorescence_empty(self):
        if not self.PL['X'] and not self.PL['Y']:
            return True
        return False





