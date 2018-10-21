#!/usr/local/bin/python3
import sys

r'''
https://www.youtube.com/watch?v=bGC2fNALbNU

Reference
* http://localhost/jforum/posts/list/2119.page
* https://www.ptt.cc/bbs/Eng-Class/M.1334215115.A.75F.html (數學的英文唸法)
'''

def find_all_subset(a_set):
    results = [[], a_set]
    
    for i in range(1, len(a_set)):
        results.extend(n_pick_m(a_set, i))

    return results

def n_pick_m(a_set, m):
    r'''
    n pick m will have  n! / (m)!*(n-m)! 

    @param a_Set(list):
        Target set with size N
    @param m(int):
        The number of M to compose the sub set

    @return:
        A list to contain all possible subset

    @see http://localhost/jforum/posts/list/2119.page
    '''
    all_sub_set_list = []
    #n_pick_m_recv(a_set, 0, m, [], all_sub_set_list)
    n_pick_m_iter(a_set, m, all_sub_set_list)

    return all_sub_set_list


class NMIter:
    def __init__(self, id_list, n):
        self.id_list = id_list
        self.n = n
        self.m = len(id_list)
        self.is_done = False
        self.jp = self.m - 2

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration()
        else:
            m = self.m
            n = self.n
            tv = self.id_list[:]
            if self.id_list[-1] + 1 < self.n:
                self.id_list[-1] += 1
            else:
                self.is_done = True

                cjp = self.jp
                while cjp >= 0:
                    if self.id_list[cjp] + 1 + (self.m - 1 - cjp) < n:
                        self.jp = cjp
                        self.is_done = False
                        self.id_list[cjp] += 1
                        npv = self.id_list[cjp] + 1
                        for k in range(cjp+1, self.m):
                            self.id_list[k] = npv
                            npv += 1

                        break
                    cjp -= 1
            return tv


def n_pick_m_iter(a_set, m, ct):
    r'''
    Iterative version of N pick M

    @param a_set(list):
        Set with size as N
    @param m(int):
        The number of M to compose the sub set
    @param ct(list):
        Collection to hold all subset
    '''
    if m == 1:
        for e in a_set:
            ct.append([e])
    elif m == len(a_set):
        ct.append(a_set)
    elif m < len(a_set):
        id_list = []
        for i in range(m):
            id_list.append(i)

        jp = m - 2
        nm_iter = NMIter(id_list, len(a_set))
        for iv in nm_iter:
            ct.append([a_set[i] for i in iv])           
    else:
        #raise Exception('M is greater than N!')
        return 

    
def n_pick_m_recv(a_set, dep, rm, ss, ct):
    r'''
    Recursive version of N pick M

    @param a_set(list):
        Set with size as N
    @param dep(int):
        Depth of process
    @param rm(int):
        Remain element from target M
    @param ss(list):
        Current sub set
    @param ct(list):
        Collection to hold all subset
    '''
    if rm == 0:
        ct.append(ss)
        return
    elif dep == len(a_set):
        return 
    else:
        # Skip current depth
        n_ss = ss[:]
        n_pick_m_recv(a_set, dep+1, rm, n_ss, ct)
 
        # Pickup element at current depth
        n_ss = ss[:]
        n_ss.append(a_set[dep])
        n_pick_m_recv(a_set, dep+1, rm-1, n_ss, ct)


def main():
    a_set = sys.argv[1:]
    print("Target set={}".format(a_set))
    results = find_all_subset(a_set)
    print("Target set={} has {:,d} subset:".format(a_set, len(results))) 
    for ss in sorted(results, key=lambda e:len(e)):
        print("{{{}}}".format(','.join(ss)))

    print("")


if __name__ == '__main__':
    main()
