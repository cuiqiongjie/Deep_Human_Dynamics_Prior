#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pprint import pprint

'''
# 平衡数据
1. walking: 12_01
2. running: 
3. wash_window
4. basketball

# 非平衡数据
boxing
gymnastics
dancing
soccer 
sweep floor
jump
closing umbrella
opening umbrella
closing a box	
opening a box
coiling a rope
Picking up Ball
swimming
eating a sandwich
'''


class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self, motion, missing_solution):
        motion = motion
        # -----------------------------------loss parameter--------------------------------------
        self.parser.add_argument("--mse_loss_weight", help="~", default=0.99, type=float)
        self.parser.add_argument("--bone_length_loss_weight", help="~", default=0.00915, type=float)
        self.parser.add_argument("--tv_loss_weight", help="~", default=0.00085, type=float)

        # -----------------------------------training parameter--------------------------------------
        self.parser.add_argument("--batch_size_", help="~", default=1, type=int)
        self.parser.add_argument("--sample_range", help="~", default=120, type=int)
        self.parser.add_argument("--checkpoint", help="specify which model to load in", default=1000, type=int)
        self.parser.add_argument("--motion_range_start", help="specify which model to load in", default=0, type=int)
        self.parser.add_argument("--motion_range_end", help="specify which model to load in", default=120, type=int)

        # -----------------------------------data parameter--------------------------------------
        if motion == "walking":
            self.parser.add_argument("--iteration", help="~", default=500, type=int)
            self.parser.add_argument("--motion_name", help="motion type", default='walking', type=str)
            self.parser.add_argument("--data_dir", help="training samples", default="./dataset/walking.mat", type=str)

        if missing_solution == "random_corruption":
            self.parser.add_argument("--corruption_method",
                                     help="randomly remove a certain proportion of specific joint information",
                                     default='random_corruption', type=str)
            self.parser.add_argument("--missing_ratio", help="~", default=.5, type=float)
            self.parser.add_argument("--missing_joints", help="~", default=3, type=int)
        elif missing_solution == "continue_corruption":
            self.parser.add_argument("--corruption_method",
                                     help="continuously remove a certain range of specific joint information",
                                     default='continue_corruption', type=str)
            self.parser.add_argument("--missing_start", help="~", default=30, type=int)
            self.parser.add_argument("--missing_end", help="~", default=60, type=int)
            self.parser.add_argument("--missing_joints", help="~", default=2, type=int)
        elif missing_solution == "gap_corruption":
            self.parser.add_argument("--corruption_method", help="remove all joints information of a certain duration",
                                     default='gap_corruption', type=str)
            self.parser.add_argument("--missing_start", help="~", default=50, type=int)
            self.parser.add_argument("--missing_end", help="~", default=60, type=int)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, motion="walking"):
        # motion = "walking"
        solution = "continue_corruption"
        self._initial(motion, solution)
        self.opt = self.parser.parse_args()
        return self.opt
