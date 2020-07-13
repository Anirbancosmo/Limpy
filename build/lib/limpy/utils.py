#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:57:35 2020

@author: anirbanroy
"""
from __future__ import division
import numpy as np
import cosmos 
import params as p

cosmo=cosmos.cosmo()

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

def boxsize_to_degree(z, boxsize):
    #boxsize in Mpc
    mpc_to_m=p.default_constants['mpc_to_m']
    boxsize=boxsize*mpc_to_m
    da=cosmo.D_angular(z)
    theta_rad=boxsize/da
    theta_degree=theta_rad*180.0/np.pi
    return theta_degree
