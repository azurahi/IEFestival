import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

def read_data(file_name):
    df = pd.read_csv(file_name, encoding = 'cp1252')
    li = list(df.columns)
    li[0] = 'ddate'
    df.columns = li
    df1 = df.iloc[::-1]
    df1.index = df1['ddate']
    del df1['ddate']
    df.index = df['ddate']
    del df['ddate']

    return df1, df

# print(df1)
def stastics(df,num):
    day =3
    data = df.diff(periods = day)
    # aaa = data.dropna().values / df1.values
    # print(aaa)
    returns = -data/df
    # returns = returns.reindex(index=returns.index[::-1])

    returns.to_csv('./returns_%s.csv'%num)

    returns.rolling(100).mean().to_csv('./mean_%s.csv'%num)
    # returns.rolling().var().to_csv('./var_%s.csv'%day)
    returns.rolling(100).cov().to_csv('./cov_%s.csv'%num)

def cal_prob(returns,date):
    # date = '2016-08-08'
    prob = pd.read_csv('./result/3-3-%s.csv'%date, encoding = 'cp1252', names=['xxx','after','before','upup','updown','downup','downdown'])
    # returns = pd.read_csv('./returns_3.csv',encoding = 'cp1252', index_col=0)
    del prob['xxx']
    n = len(returns.ix[0])
    prob = prob.drop(0)
    theday = np.where(returns.index==date)[0]
    change = np.sign(returns.ix[theday[0]+100])

    before_prob = np.zeros((2,len(change)))
    before_prob[0,np.where(change.values == 1)]=1
    before_prob[1,np.where(change.values != 1)]=1

    before_prob = np.tile(before_prob,n)
    after_prob = prob.iloc[:,2:4].values * before_prob.T
    p =np.zeros((n,1))
    for i in range(n):
        p[i] = after_prob[n*i:n*i+n].mean()*2
    return p

def markov_portfolio(returns,means, cov, date ='2015-10-07'):
    n = len(returns.ix[0])
    index = returns.index
    theday = index[np.argwhere(index==date)+100].values[0,0]
    ret = returns.loc[theday].values.astype(np.double)
    r_mean = means.loc[theday].values.astype(np.double).mean()

    P = opt.matrix(cov.loc[theday].values.astype(np.double))
    q = opt.matrix(np.zeros((n, 1)), tc='d')
    G = -opt.matrix(np.concatenate((
        np.array([ret]),
        np.eye(n)), 0))
    h = opt.matrix(np.concatenate((
        -np.ones((1, 1)) * r_mean,
        np.zeros((n, 1))), 0))
    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)['x']
    return np.asarray(sol)



def random_porfolio(returns, date):
    n = len(returns.ix[0])
    k = np.random.rand(n)
    return k / sum(k)

def optimal_portfolio(returns,means, cov, date ='2015-10-07'):
    n = len(returns.ix[0])
    index = returns.index
    theday = index[np.argwhere(index == date) + 100].values[0, 0]
    ret = returns.loc[theday].values.astype(np.double)
    r_mean = means.loc[theday].values.astype(np.double).mean()

    prob = cal_prob(returns, date)
    prob_mean = prob.mean()
    print(prob)
    print(prob_mean)

    P = opt.matrix(cov.loc[theday].values.astype(np.double))
    q = opt.matrix(np.zeros((n, 1)), tc='d')
    G = -opt.matrix(np.concatenate((
        # np.array([ret]),
        np.array(prob.T),
        np.eye(n)), 0))
    h = opt.matrix(np.concatenate((
        # -np.ones((1, 1)) * r_mean,
        -np.ones((1, 1)) * prob_mean,
        np.zeros((n, 1))), 0))
    # equality constraint Ax = b; captures the constraint sum(x) == 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)['x']
    return np.asarray(sol)

    # return None

def prob_portfolio(returns,cov,date, N):
    prob = cal_prob(returns,date)
    val = np.argsort(prob.T)
    # wt = np.zeros((1,len(returns.ix[0])))

    p = np.where(val <N)[1].tolist()
    print(p)
    idx=[]

    a = returns.columns[p].values

    n = len(returns.ix[0])
    index = returns.index
    theday = index[np.argwhere(index==date)+100].values[0,0]
    ret = returns.loc[theday, a].values.astype(np.double)
    r_mean = means.loc[theday,a].values.astype(np.double).mean()
    try:
        P = opt.matrix(cov.loc[theday,a].loc[a,:].values.astype(np.double))
        q = opt.matrix(np.zeros((N, 1)), tc='d')
        G = -opt.matrix(np.concatenate((
            np.array([ret]),
            np.eye(N)), 0))
        h = opt.matrix(np.concatenate((
            -np.ones((1, 1)) * r_mean,
            np.zeros((N, 1))), 0))
        # equality constraint Ax = b; captures the constraint sum(x) == 1
        A = opt.matrix(1.0, (1, N))
        b = opt.matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)['x']
        wt = np.zeros((1, n))

        wt[:,p] = np.asarray(sol).T
    except:
        wt = np.zeros((1, n))
        wt[:,p] = np.asarray([0.2]*5)
    return wt

if __name__ == '__main__':
    for i in range(5):
        df1, df = read_data('./data/%s.csv'%i)
        # stastics(df, i)

        len_day = df.shape[0]
        num_stock = df.shape[1]

        means = pd.read_csv('./mean_%s.csv'%i, index_col=0, encoding='cp1252')
        returns = pd.read_csv('./returns_%s.csv'%i, index_col=0, encoding='cp1252')
        cov = pd.read_csv('./cov_%s.csv'%i, index_col=[0, 1], encoding='cp1252')

        # print(returns.index)
        # print(returns.columns.values)

        balancing_day = returns.index[0:len_day-100]
        # df = returns.copy()
        # print(df.values)
        df = pd.DataFrame(np.zeros((len_day,num_stock)), index= returns.index, columns=returns.columns.values )
        print(df)
        m_df = df.copy()
        o_df = df.copy()
        w_df = df.copy()
        r_df = df.copy()
        for j in range(int((len_day-100)/3)):
            try:
                date = balancing_day[3*j]
                theday = returns.index[np.argwhere(returns.index == date) + 100].values[0, 0]
                print(theday)
                m_df.loc[theday] = markov_portfolio(returns, means, cov, date).T.tolist()[0]
                # o_df.loc[theday] = optimal_portfolio(returns, means, cov, date).T.tolist()[0]
                w_df.loc[theday] = prob_portfolio(returns, cov,date, 5)
                r_df.loc[theday] = random_porfolio(returns,date)
            except:
                None
        m_df.to_csv('./port/markov_port_%s.csv'%i)
        # o_df.to_csv('./port/optimal_port_%s.csv'%i)
        w_df.to_csv('./port/prob_port_%s.csv'%i)
        r_df.to_csv('./port/random_port_%s.csv'%i)
# wt = optimal_portfolio()

