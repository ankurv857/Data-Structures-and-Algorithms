#author @ankurverma
#Data Structures & Algorithms class implementation

def MissingNumber(array,n):
    array = sorted(array)
    for i in range(1,len(array)):
        if array[i] != array[i-1] + 1:
            return array[i-1] + 1
    if array[0] == 2:
        return array[0] - 1
    else:
        return array[-1] + 1

import collections
def majorityElement(A,N):
    d = collections.Counter(A)
    for key in d:
        if d[key] > N//2:
            return key
    return -1

