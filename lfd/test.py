#!/usr/bin/env python
import sys
import recognition
import numpy as np

def main():
	xyz = np.array([[0.1,0.1,0.1],[0.5,0.5,0.5],[0.7, 0.7, 0.7]])
	print xyz
	print type(xyz)
	print xyz.shape
	recognition.calc_geodesic_distances(xyz)
main()
