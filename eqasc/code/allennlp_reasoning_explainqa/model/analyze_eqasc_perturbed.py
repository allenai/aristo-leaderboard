import numpy as np
import sys

def compare(fname1, fname2):
    data1 = open(fname1,'r').readlines()
    header1, data1 = data1[0], data1[1:]
    data2 = open(fname2, 'r').readlines()
    header2, data2 = data2[0], data2[1:]
    assert len(data1) == len(data2)
    data1 = [row.strip().split('\t') for row in data1]
    data2 = [row.strip().split('\t') for row in data2]
    # fact1	fact2	combined
    # fact1edited	fact2edited	combinededited
    abs_differences = []
    for r1,r2 in zip(data1,data2):
        abs_differences.append( np.abs(float(r1[-1].strip()) - float(r2[-1].strip())) )
    abs_differences = np.array(abs_differences)
    print("# total: ", len(abs_differences) )
    print("# abs_change==0: ", len([d for d in abs_differences if d<=0]))


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
