#!/usr/local/bin/python3
r'''
https://www.hackerrank.com/challenges/common-child/problem

Ref:
* https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
'''
import math
import os
import random
import re
import sys
import pytest

from pprint import pprint

# Complete the commonChild function below.
def commonChild(s1, s2):
    m = len(s1)
    n = len(s2)
    caches = [[0] * (n+1) for i in range(m+1)]    
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                caches[i+1][j+1] = caches[i][j] + 1
            else:
                caches[i+1][j+1] = max(caches[i+1][j], caches[i][j+1])

    return caches[m][n]

def commonChild_v4(s1, s2):
    r'''
    Skip collect string(s) from LCS from v3    
    '''
    caches = {}
    lcs_max = 0
    for i, c1 in enumerate(s1):
        for j, c2 in enumerate(s2):
            if c1 == c2:
                if i == 0 and j == 0:
                    caches[(i, j)] = 1
                    if lcs_max == 0:
                        lcs_max = 1
                elif i > 0 and j == 0:
                    caches[(i, j)] = 1
                    if lcs_max == 0:
                        lcs_max = 1
                elif j > 0 and i == 0:
                    caches[(i, j)] = 1
                    if lcs_max == 0:
                        lcs_max = 1
                else:
                    if caches[(i-1, j-1)]:
                        tv = caches[(i-1, j-1)] + 1
                    else:
                        tv = 1
                
                    caches[(i, j)] = tv

                if caches[(i, j)] > lcs_max:
                    lcs_max = caches[(i, j)]

            else:
                if i == 0 and j == 0:
                    caches[(0, 0)] = 0
                elif i > 0 and j == 0:
                    caches[(i, j)] = caches[(i-1, j)]
                elif j > 0 and i == 0:
                    caches[(i, j)] = caches[(i, j-1)]
                else:
                    cvi = caches[(i-1, j)]
                    cvj = caches[(i, j-1)]

                    caches[(i, j)] = max(cvi, cvj)

        if i and i % 10 == 0:
            # Clean caches to avoid out of memory
            caches = dict(filter(lambda t: i - t[0][0]==0, caches.items()))

    return lcs_max

def commonChild_v3(s1, s2):
    caches = {}
    lcs_max = 0
    lcs_str = None
    for i, c1 in enumerate(s1):
        for j, c2 in enumerate(s2):
            if c1 == c2:
                if i == 0 and j == 0:
                    caches[(i, j)] = [(c1,)]
                    if lcs_max == 0:
                        lcs_max = 1
                        lcs_str = c1
                elif i > 0 and j == 0:
                    caches[(i, j)] = [(c1,)]
                    if lcs_max == 0:
                        lcs_max = 1
                        lcs_str = c1
                elif j > 0 and i == 0:
                    caches[(i, j)] = [(c1,)]
                    if lcs_max == 0:
                        lcs_max = 1
                        lcs_str = c1
                else:
                    if caches[(i-1, j-1)]:
                        t_list = []
                        for cv_t in caches[(i-1, j-1)]:
                            nt = tuple(list(cv_t) + [c1])
                            t_list.append(nt)
                            if len(nt) > lcs_max:
                                lcs_max = len(nt)
                                lcs_str = ''.join(nt)

                        caches[(i, j)] = t_list
                    else:
                        caches[(i, j)] = [(c1,)]        

            else:
                if i == 0 and j == 0:
                    caches[(0, 0)] = []
                elif i > 0 and j == 0:
                    caches[(i, j)] = caches[(i-1, j)]
                elif j > 0 and i == 0:
                    caches[(i, j)] = caches[(i, j-1)]
                else:
                    cvi = caches[(i-1, j)]
                    cvj = caches[(i, j-1)]
                    
                    cv_set = set(cvi) | set(cvj)
                    caches[(i, j)] = list(cv_set)
            
    print('LCS got {} ({:})'.format(lcs_str, lcs_max))
    return lcs_max

def commonChild_v2(s1, s2):
    r'''
    Recursive version of LCS
    '''
    lcs_str =  LCS(s1, s2, '')
    print('LCS got {} ({:})'.format(lcs_str, len(lcs_str)))
    return len(lcs_str)

lcs_cache = {}
def LCS(s1, s2, px):
    global lcs_cache
    # Short cut
    if len(s1) == 0 or len(s2) == 0:
        return px
    elif s1 == s2:
        return s1 + px

    # Memorization
    if (s1, s2) in lcs_cache:
        return lcs_cache[(s1, s2)] + px

    # Searching
    if s1[-1] == s2[-1]:
        return LCS(s1[:-1], s2[:-1], s1[-1] + px)
    else:
        sub_sol1 = LCS(s1[:-1], s2, px)
        sub_sol2 = LCS(s1, s2[:-1], px)
        if len(sub_sol1) >= len(sub_sol2):
            lcs_cache[(s1, s2)] = sub_sol1[:-(len(px))] if px else sub_sol1
            lcs_cache[(s2, s1)] = sub_sol1[:-(len(px))] if px else sub_sol1
            return sub_sol1
        else:
            lcs_cache[(s1, s2)] = sub_sol2[:-(len(px))] if px else sub_sol2
            lcs_cache[(s2, s1)] = sub_sol2[:-(len(px))] if px else sub_sol2
            return sub_sol2

# Complete the commonChild function below.
def commonChild_v1(s1, s2):
    # Short cut
    if len(s1) == 0 or len(s2) == 0:
        return 0
    elif s1 == s2:
        return len(s1)

    # dict to keep first chr point        
    s1c_dict = {}
    s2c_dict = {}
    
    for s, d in [(s1, s1c_dict), (s2, s2c_dict)]:
        ci = 0
        for c in s:
            if c not in d:
                d[c] = s[ci:]

            ci += 1
    
    # Start searching for child with max length
    hit_c_set = set(s1c_dict.keys()) & set(s2c_dict.keys())
    max_len = 0
    max_sub = ''
    for c in hit_c_set:
        s1s = s1c_dict[c]
        s2s = s2c_dict[c]

        msm = searchMSM(s2s, s1s)

        sub_max = len(msm)
        if sub_max > max_len:
            max_len = sub_max
            max_sub = msm

    print("\t[Info] Got '{}' ({:,d})".format(max_sub, max_len))
        
    return max_len


def commonChild_v0(s1, s2):
    msm1 = searchMSM(s1, s2)
    msm2 = searchMSM(s2, s1)
    print("\t[Info] Got {}\t{}".format(msm1, msm2))
    return max(len(msm1), len(msm2))


def searchMSM(s1, s2):
    msm_cache = {}
    return searchMSM_recv(s1, s2, '', msm_cache)

miss_set = set()
def searchMSM_recv(s1, s2, ms, msm_cache):
    global miss_set

    if len(s1) == 0 or len(s2) == 0:
        return ms
    elif s1 == s2:
        return ms + s1
    elif (s1, s2) in miss_set or (s2, s1) in miss_set:
        return ms

    oc_set = set(s1) & set(s2)
    if len(oc_set) == 0:
        return ms 

    # Shorten the input strings
    #s1 = ''.join(filter(lambda c: c in oc_set, list(s1)))
    #s2 = ''.join(filter(lambda c: c in oc_set, list(s2)))

    # Cache block
    cr = msm_cache.get((s1, s2), None)
    if cr is not None:
        return ms + cr

    ci = 0
    for c in s1:
        try:
            mp = s2.index(c)
            nms = ms + c
            sms1 = searchMSM_recv(s1[ci+1:], s2[mp+1:], nms, msm_cache)
            msm_cache[(s1[ci+1:], s2[mp+1:])] = sms1[len(nms):]
            sms2 = searchMSM_recv(s1[ci+1:], s2, ms, msm_cache)
            msm_cache[(s1[ci+1:], s2)] = sms2[len(ms):]
            if len(sms1) >= len(sms2):
                #print("\t{}\t{}\t{}>{} ({:d})".format(ms, s1, s2, sms1, len(sms1)))
                return sms1
            else:
                #print("\t{}\t{}\t{}>{} ({:d})".format(ms, s1, s2, sms2, len(sms2)))
                return sms2
        except:
            pass

        ci += 1

    miss_set.add((s1, s2))
    return ms


def searchMSM_v1(s1, s2):
    r'''
    Search for maximum sequence match

    @param s1(str):
        Target string1 
    @param s2(str):
        Matched string

    @return
        Look for the lenght of maximum sequence matched string
    '''
    hc = ''
    for c in s1:
        try:
            mp = s2.index(c)
            if mp >= 0:
                hc += c
                s2 = s2[mp+1:] 
        except:
            pass

        if not s2:
            break

    return hc

import unittest


class FAT(unittest.TestCase):
    @staticmethod
    def read_tc(tn):
        if __file__.startswith('./'):
            script_fn = __file__[2:]
        else:
            script_fn = __file__
        tf_name = '{}.t{}'.format(script_fn.split('.')[0], tn)
        with open(tf_name, 'r') as fh:
            s1 = fh.readline().strip()
            s2 = fh.readline().strip()
            a = int(fh.readline().strip())

        return (s1, s2, a)

    def test_demo(self):
        for s1, s2, a in [
                            ('HARRY', 'SALLY', 2),
                            ('AA', 'BB', 0),
                            ('SHINCHAN', 'NOHARAAA', 3),
                            ('ABCDEF', 'FBDAMN', 2)
                         ]:
            r = commonChild(s1, s2) 
            self.assertEqual(a, r, "Expect={}; Real={} on '{}' vs '{}'".format(a, r, s1, s2))

    #@pytest.mark.skip(reason="Take a long time")
    def test_real(self):
        for tn in [1, 2, 3, 4, 5]:
            s1, s2, a  = FAT.read_tc(tn)
            r = commonChild(s1, s2)
            self.assertEqual(a, r, "Expect={}; Real={} on '{}' vs '{}' (tn={})".format(a, r, s1, s2, tn))

def main(s1, s2):
    return commonChild(s1, s2)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(main(sys.argv[1], sys.argv[2]))
    else:
        s1, s2, a = FAT.read_tc(sys.argv[1])
        print(commonChild(s1, s2))

