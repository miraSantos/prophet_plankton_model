import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.forecasting_metrics import *
sns.set()


length = 5000
period = 365
noise_variance = 0.2

ds = pd.date_range(datetime.today(), periods=length).tolist()
temp = np.sin(2*np.pi/period*np.linspace(0,length,length)) + 10
noise = np.random.normal(0,noise_variance,length)
dependent = np.cos(temp + 0.3) + 20000 + noise

fig, ax = plt.subplots(2,1,figsize = (14,16))
ax[0].scatter(ds,temp, label="temp",c = "red")
ax[0].set_title("Temperature")
ax[1].scatter(ds,dependent, label="dependent")
ax[1].set_title("Dependent Variable, noise variance = "+ str(noise_variance))
plt.show()

plt.savefig("/home/mira/PycharmProjects/prophet_plankton_model/datasets/synthetic/temp_conc_10Dec2022.png")

df = pd.DataFrame(list(zip(ds, temp, dependent)), columns =['date', 'temp','dependent'])

print(df.head())

save_path = "/home/mira/PycharmProjects/prophet_plankton_model/datasets/synthetic"
df.to_csv(save_path + "/temp_conc_10Dec2022.csv")