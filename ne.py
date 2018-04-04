import numpy as np
import pandas as pd
import datetime, itertools
from multiprocessing import Pool

def readData():
    df = pd.read_csv('./merged.csv', encoding = 'cp1252')
    li =list(df.columns)
    li[0] = 'ddate'
    df.columns = li
    df1 = df.iloc[::-1]
    df1.index = df1['ddate']
    del df1['ddate']
    df.index = df['ddate']
    del df['ddate']
    return df1, df

def findDiff(df, days = 3):
    print(np.sign(df.diff(periods = days)))
    return np.sign(df.diff(periods = days))

def conditionalProb(s1, s2):
    try:
        s1.dropna(inplace=True)
        s2.dropna(inplace=True)
        commonIx = np.intersect1d(s1.index, s2.index)
        s1, s2 = s1.loc[commonIx], s2.loc[commonIx]
        s1up, s1down, s1con = s1[s1>0].index,s1[s1<=0].index,s1[s1==0].index
        s2up, s2down, s2con = s2[s2 < 0].index, s2[s2 >= 0].index, s2[s2 == 0].index
        upup = len(np.intersect1d(s1up, s2up))/len(s1up)
        #upcon = len(np.intersect1d(s1up, s2con))/len(s1up)
        updown = len(np.intersect1d(s1up, s2down)) / len(s1up)
        #conup = len(np.intersect1d(s1con, s2up)) / len(s1con)
        #concon = len(np.intersect1d(s1con, s2con)) / len(s1con)
        #condown = len(np.intersect1d(s1con, s2down)) / len(s1con)
        downup = len(np.intersect1d(s1down, s2up)) / len(s1down)
        #downcon = len(np.intersect1d(s1down, s2con)) / len(s1down)
        downdown = len(np.intersect1d(s1down, s2down)) / len(s1down)
        return [s1.name, s2.name, upup, updown, downup, downdown]

    except:
        return []

def makeConditionalProbTable(chunk):
    dfNormal= chunk[0].loc[chunk[-1][0]]
    dfReversed = chunk[1].loc[chunk[-1][1]]
    a = chunk[2]
    b = chunk[3]
    afterCriteria = findDiff(dfNormal,days = a)
    previousCriteria = findDiff(dfReversed,days = b )
    li = []
    for afterCol in afterCriteria.columns[:]:
        for prevCol in previousCriteria.columns:
            li.append(conditionalProb(afterCriteria[afterCol],previousCriteria[prevCol]))
    result  = pd.DataFrame(li)
    result.to_csv('./result/%s-%s-%s.csv'%(str(a), str(b), str(dfNormal.index[0])))
    """
    except:
        print(a, b, ' have an error')
        return [a,b,dfNormal.index[0]]
    """
    print(a, b, ' is done',datetime.datetime.now())

def makeConProbMulti(dfNormal, dfReversed, mina, increda, minb,incredb, NProc = 4):
    days = [[dfNormal.index[0+day:100+day], dfReversed.index[500-100-day:500-day]] for day in np.arange(389)]
    l=[[i, j, k] for i, j, k in itertools.product(np.arange(increda)+mina, np.arange(incredb)+minb, days)]
    chunks = [[dfNormal,dfReversed,i,j, k] for i, j, k in itertools.product(np.arange(increda)+mina, np.arange(incredb)+minb, days)]
    pool = Pool(processes=NProc)
    list = pool.map(makeConditionalProbTable, chunks)
    print(list)
    pool.terminate()

if __name__=='__main__':
    dfNormal, dfReversed=readData()
    makeConProbMulti(dfNormal, dfReversed,3, 7, 3,7, NProc=4)