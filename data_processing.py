 # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pymysql, queue

def readData(stockCode):
    db = pymysql.connect('localhost', 'root', 'autoset', 'financialdata')
    cursor = db.cursor()
    sql = "select ddate, close from krx%s where ddate>='2015-09-22' order by ddate desc limit 500;" %stockCode
    cursor.execute(sql)
    df = cursor.fetchall()
    df = pd.DataFrame(np.array(df))
    db.close()
    df.set_index(df.columns[0], inplace=True)
    df.columns = ['krx'+str(stockCode)]
    df.index.name = None
    print(len(df))
    return df
def mergeData(dfList):
    a = queue.Queue()
    for i in dfList:
        a.put(i)
    while(True):
        temp = a.get()
        temp2 = a.get()
        if(len(temp)==500 and len(temp2)==500):
            tp = pd.concat([temp,  temp2], axis = 1)
            a.put(tp)
        elif(len(temp)<500 and len(temp2)==500):
            a.put(temp2)
        elif(len(temp)==500 and len(temp2)<500):
            a.put(temp)
        print(a.qsize())
        if(a.qsize()<=1):
            break
    return a.get()
def indicesToCsv():
    db = pymysql.connect('localhost', 'root', 'autoset', 'financialdata', charset='utf8')
    cursor = db.cursor()
    sql = "select ddate,'코스피종가', '코스닥종가', 'DJIDJIclos', 'DJIDJTclos', 'NASIXICclos', 'NASNDXclos', 'SPISPXclos', 'NASSOXclos', 'NIINI225clos', 'BRIBVSPclos', 'SHS000001clos', 'SHS000002clos', 'SHS000003clos', 'HSIHSIclos', 'HSIHSCEclos', 'HSIHSCCclos', 'TWSTI01clos', 'INIBSE30clos', 'MYIKLSEclos', 'IDIJKSEclos', 'LNSFTSE100clos', 'PASCAC40clos', 'XTRDAX30clos', 'STXSX5Eclos', 'RUIRTSIclos', 'ITIFTSEMIBclos' from indices order by ddate desc limit 500 ;".replace(
        "\'", "")
    cursor.execute(sql)
    df = pd.DataFrame(np.array(cursor.fetchall()))
    df.columns = ['ddate', '코스피종가', '코스닥종가', 'DJIDJIclos', 'DJIDJTclos', 'NASIXICclos', 'NASNDXclos', 'SPISPXclos',
                  'NASSOXclos', 'NIINI225clos', 'BRIBVSPclos', 'SHS000001clos', 'SHS000002clos', 'SHS000003clos',
                  'HSIHSIclos', 'HSIHSCEclos', 'HSIHSCCclos', 'TWSTI01clos', 'INIBSE30clos', 'MYIKLSEclos',
                  'IDIJKSEclos', 'LNSFTSE100clos', 'PASCAC40clos', 'XTRDAX30clos', 'STXSX5Eclos', 'RUIRTSIclos',
                  'ITIFTSEMIBclos']
    db.close()
    df.to_csv('./temp/indices.csv')
def resourcesToCsv():
    """
    marketindexCd = ['CMDT_HO', 'CMDT_NG', 'CMDT_CDY', 'CMDT_PDY', 'CMDT_ZDY', 'CMDT_AAY', 'CMDT_SDY', 'CMDT_C'
        , 'CMDT_SB', 'CMDT_S', 'CMDT_SM', 'CMDT_BO', 'CMDT_CT', 'CMDT_W', 'CMDT_RR', 'CMDT_OJ', 'CMDT_KC',
                     'CMDT_CC']
    cols = [i+'close' for i in marketindexCd]"""
    sql = "select ddate, 'CMDT_HOclose', 'CMDT_NGclose', 'CMDT_CDYclose', 'CMDT_PDYclose', 'CMDT_ZDYclose', 'CMDT_AAYclose', 'CMDT_SDYclose', 'CMDT_Cclose', 'CMDT_SBclose', 'CMDT_Sclose', 'CMDT_SMclose', 'CMDT_BOclose', 'CMDT_CTclose', 'CMDT_Wclose', 'CMDT_RRclose', 'CMDT_OJclose', 'CMDT_KCclose', 'CMDT_CCclose' from aresources order by ddate desc limit 500;".replace("'","")
    db = pymysql.connect('localhost', 'root', 'autoset', 'financialdata', charset='utf8')
    cursor = db.cursor()
    cursor.execute(sql)
    df = pd.DataFrame(np.array(cursor.fetchall()))
    df.columns = ['ddate','CMDT_HOclose', 'CMDT_NGclose', 'CMDT_CDYclose', 'CMDT_PDYclose', 'CMDT_ZDYclose', 'CMDT_AAYclose', 'CMDT_SDYclose', 'CMDT_Cclose', 'CMDT_SBclose', 'CMDT_Sclose', 'CMDT_SMclose', 'CMDT_BOclose', 'CMDT_CTclose', 'CMDT_Wclose', 'CMDT_RRclose', 'CMDT_OJclose', 'CMDT_KCclose', 'CMDT_CCclose']
    db.close()
    df.to_csv('./temp/resources.csv')
    print(df)

if __name__=='__main__':
    resourcesToCsv()

