import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
copy = lambda obj: pickle.loads(pickle.dumps(obj))
def relax(main, other): return main if main > other else other
class Result:
    def __init__(self, balance):
        self.balance = balance
        self.volume = 0
        self.prevPosIndex = -1
        self.openIndex = 0
        self.closeIndex = 0

    def __gt__(self, other): 
        return self.balance > other.balance  
data = pd.read_csv("new.csv")
data = data[["date", "time", "price"]]
class Investments:
    def __init__(self, data, transacMax, startBalance):
        self.data = data.copy()
        self.transacMax = transacMax
        self.startBalance = startBalance
        
    def fit(self):
        dp = np.full((self.transacMax + 1,2), Result(0)) # Для подсчета используется динамика dp[i][flg] 
        # с восстановлением ответа. dp[i][flg] - макс результат, если мы еще можем совершить i сделок,
        # flg = 1, если при этом мы находимся в позиции, flg = 0 иначе
        dp[self.transacMax][0] = Result(self.startBalance)

        prevPrice = self.data["price"][0]
        self.positions = []

        for index, price in enumerate(self.data["price"]):
            for j in range(self.transacMax + 1):
                dp[j][1].balance += dp[j][1].volume * (price - prevPrice)
                
            for j in range(1, self.transacMax + 1):
                pos = copy(dp[j][1])
                pos.closeIndex = index
                self.positions.append(copy(pos))
                pos.prevPosIndex = len(self.positions)-1
                dp[j - 1][0] = copy(relax(dp[j - 1][0], pos))

            for j in range(1, self.transacMax + 1):
                pos = copy(dp[j][0])
                pos.volume = dp[j][0].balance // price
                pos.openIndex = index
                dp[j][1] = copy(relax(dp[j][1], pos))

            prevPrice = price
            
        self.indexesOfPos = [dp[0][0].prevPosIndex]
        for j in range(self.transacMax-1): # Восстановление ответа
            self.indexesOfPos.append(self.positions[self.indexesOfPos[-1]].prevPosIndex)
            
    def show(self, pictureName = None):
        fig, ax = plt.subplots(figsize=(32,16))
        ax.plot(self.data["price"], linewidth=1)
    
        for indexOfPos in self.indexesOfPos:
            openIndex = self.positions[indexOfPos].openIndex
            closeIndex = self.positions[indexOfPos].closeIndex    
            ax.plot([openIndex, closeIndex], [self.data["price"][openIndex], self.data["price"][closeIndex]],
                    linewidth=6)

        ax.set(xlabel='time', ylabel='price')
        ax.grid()
        if not(pictureName is None):
            fig.savefig(pictureName)
        plt.show()
        
    def getInfo(self, fileNameCSV = None):
        for number, indexOfPos in enumerate(self.indexesOfPos[::-1]):
            openIndex = self.positions[indexOfPos].openIndex
            closeIndex = self.positions[indexOfPos].closeIndex
            cInfo = self.data[openIndex+1: closeIndex +1].copy()
            cInfo["price"] = np.array(cInfo["price"]) - np.array(self.data[openIndex:closeIndex]["price"])
            cInfo.index = range(closeIndex-openIndex)
            cInfo["date_change"] = cInfo["date"]
            cInfo["time_change"] = cInfo["time"]
            cInfo["price_change"] = cInfo["price"]
            cInfo.drop(["price","date","time"], inplace = True, axis = 1)
            mainInfo = pd.DataFrame({'date_open'  :[self.data["date"][openIndex]],
                                     'time_open'  :[self.data["time"][openIndex]],
                                     'price_open' :[self.data["price"][openIndex]],
                                     'time_close' :[self.data["date"][closeIndex]],
                                     'date_close' :[self.data["date"][closeIndex]],
                                     'price_close':[self.data["price"][closeIndex]],
                                     'volume'     :[self.positions[indexOfPos].volume]})
            df = cInfo.join(mainInfo)
            if not(fileNameCSV is None):
                df.to_csv(str(number) + '_' + fileNameCSV)
        
            print("Позиция № ",number + 1, " была открыта ", self.data["date"][openIndex], " в ",
                  self.data["time"][openIndex], " по цене ", self.data["price"][openIndex],
                  " объемом ", self.positions[indexOfPos].volume)
            print("Позиция № ",number +  1, " была закрыта ", self.data["date"][closeIndex], " в ",
                  self.data["time"][closeIndex], " по цене ", self.data["price"][closeIndex])
            print("Баланс ", self.positions[indexOfPos].balance)
            
  
print("Введите максимальное количество транзакций")
K = int(input())
print("Введите начальный депозит")
Balance = float(input())    
model = Investments(data, K, Balance)
model.fit()
model.getInfo("info_"+str(K)+"_"+str(Balance)+".csv")
model.show("chart_"+str(K)+"_"+str(Balance)+".png")
