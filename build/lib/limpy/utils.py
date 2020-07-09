#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:57:35 2020

@author: anirbanroy
"""
from __future__ import division
import numpy as np
import cosmolopy


def volume_box(boxsize):
    """
    Total volume of the simulation box. 
    unit: boxsize in Mpc
    return: volume in Mpc^3
    """
    return boxsize**3 #in mpc

def volume_cell(boxsize,ngrid):
    """
    The volume of a cell in a simulation box.
    unit: boxsize in Mpc, ngrid: 1nt
    return: volume in Mpc^3
    """
    clen= boxsize/ngrid   #length of a cell
    return clen**3   # in Mp

