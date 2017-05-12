# -*- coding: utf-8 -*-
#!/usr/bin/env python2

"""
Created on Wed May 10 19:13:33 2017

@author: ra

"""


def write2txt(d):
    result=open('./result.txt','a')

    for k,v in sorted (d.items()):
        result.write(k)
        result.write(u'     ')
        if v['sex']=='male':
            result.write(u'    男     ')
        else:
            result.write(u'    女     ')
        if v["glasses"]=="黑色":
            result.write(u'   有        黑色  ')
        elif v["glasses"]=="none":
            result.write(u'   无         无   ')
        else:
            result.write(u'   有        透明  ')
        if v["mask"]=='none':
            result.write(u'      无          无   ')
        else:
            result.write(u'      有         ')
            result.write(v["mask"])
            result.write(' ')
        if v["hat"]=='none':
            result.write(u'      无         无    ')
        else:
            result.write(u'      有        ')
            result.write(v["hat"])
        result.write('\n')
    result.close()

