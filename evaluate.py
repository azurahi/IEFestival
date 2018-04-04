import pandas as pd
import numpy as np
import portfolio as pf

if __name__ == '__main__':
    for i in range(5):
        means = pd.read_csv('./mean_%s.csv'%i, index_col=0, encoding='cp1252')
        returns = pd.read_csv('./returns_%s.csv'%i, index_col=0, encoding='cp1252')

        mdf = pd.read_csv('./port/markov_port_%s.csv'%i, index_col=0)
        # odf= pd.read_csv('./port/optimal_port_%s.csv'%i, index_col=0)
        wdf = pd.read_csv('./port/prob_port_%s.csv'%i, index_col=0)
        rdf = pd.read_csv('./port/random_port_%s.csv'%i, index_col=0)

        m_returns = returns.values * mdf.values
        # o_returns = returns.values * odf.values
        w_returns = returns.values * wdf.values
        r_returns = returns.values * rdf.values

        m_returns = m_returns.sum(axis=1) + np.ones((1,1))
        # print(returns.index.values)
        # o_returns = o_returns.sum(axis=1) + np.ones((1,1))
        w_returns = w_returns.sum(axis=1) + np.ones((1,1))
        r_returns = r_returns.sum(axis=1) + np.ones((1,1))


        m_ret = pd.DataFrame(m_returns)
        # o_ret = pd.DataFrame(o_returns)
        w_ret = pd.DataFrame(w_returns)
        r_ret = pd.DataFrame(r_returns)

        m_ret.to_csv('./port/r_markov_port%s.csv'%i)
        # o_ret.to_csv('./port/r_optimal_port%s.csv'%i)
        w_ret.to_csv('./port/r_prob_port%s.csv'%i)
        r_ret.to_csv('./port/r_random_port%s.csv'%i)

