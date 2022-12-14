import os, sys
sys.path.append(os.getcwd())
from utils.forecasting_metrics import *
import argparse
import yaml
import pandas as pd
from prophet import Prophet
import wandb  # library for tracking and visualization
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import seaborn as sns

import numpy as np


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mae', 'mse', 'smape', 'umbrae')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))



def main_sweep(config):
    train_end = len(dfsubset["y"])*config["train_size"]
    print("main_sweep")
    print(dfsubset[["ds","y"]])
    m = Prophet(weekly_seasonality=False,daily_seasonality=False,changepoint_prior_scale=0.8, growth="flat")
    m.add_seasonality(name='season', period=90, fourier_order=8)

    if config["temp"]:
        m.add_regressor('temp')
        m.fit(dfsubset.loc[:train_end,["ds","y","temp"]])
        future = m.make_future_dataframe(periods=len(dfsubset.ds), include_history=True)
        future = future.merge(dfsubset.loc[:len(future), ["ds", "temp"]], on="ds", how="inner")

    if config["temp"] == False:
        m.fit(dfsubset.loc[:train_end,["ds","y","cap"]])
        future = m.make_future_dataframe(periods=len(dfsubset.ds), include_history=True)
        future['cap'] = cap

    forecast = m.predict(future)
    print(forecast.head())
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    new = dfsubset.merge(forecast, on="ds",how= "inner")
    if config["take_exp"]:
        new.y = np.exp(new.y)
        new.yhat = np.exp(new.yhat)
    print("new")
    print(new.head())

    # ACTUAL VS PREDICTED SECTION
    actual = new.loc[train_end:, "y"]
    predicted = new.loc[train_end:, "yhat"]


    #FULL TIMESERIES and DOY Plot
    width = 20
    height = 8
    fig, ax = plt.subplots(1,2, figsize=(width, height))
    ax[0].scatter(dfsubset.loc[0:train_end,"ds"], dfsubset.loc[0:train_end,"y"], label="Observations", c="blue",marker = "o", facecolors='none')
    ax[0].plot(forecast["ds"], forecast["yhat"], label="prediction", c="red")
    ax[0].scatter(dfsubset.loc[train_end:,"ds"],dfsubset.loc[train_end:, "y"], c="mediumseagreen", marker="+", label="Testing data")
    ax[0].set_title("Prophet forecast: " + str(config["dependent"]))
    ax[0].set_ylabel(config["dependent"])
    ax[0].set_xlabel("Year")
    ax[0].legend()
    ax[0].grid()
    ax[1].scatter(new.loc[train_end:,"doy_numeric"], actual, c="mediumseagreen", label="Observations")
    ax[1].scatter(new.loc[train_end:,"doy_numeric"], predicted, c="red", label="Predictions")
    ax[1].set_ylabel(config["dependent"])
    ax[1].set_xlabel("Day of Year")
    ax[1].set_title("Prophet forecast by day of year: " +  str(config["dependent"]))
    ax[1].legend()
    ax[1].grid()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/full_timeseries_train_size_" + str(config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Full Timeseries Evaluation": wandb.Image(im)})


    #Actual vs. Predicted and Violin Plot
    new_long = pd.melt(new[["month", "ds", "y", "yhat"]].loc[train_end:],
                  id_vars=["month", "ds"], value_vars=["y","yhat"], value_name="conc")

    fig, ax = plt.subplots(1,2,figsize=(15, 5))
    ax[0].scatter(new.loc[train_end:,"ds"],actual, c="mediumseagreen", marker= "x",label="observations")
    ax[0].plot(new.loc[train_end:,"ds"],predicted, c="red", label="predictions")
    ax[0].legend()
    ax[0].set_ylabel(config["dependent"])
    ax[0].set_xlabel("Year")
    ax[0].set_title("Observations vs. Predictions for Testing")
    pal = {"y": "mediumseagreen", "yhat": "r"}
    sns.violinplot(axes=ax[1],data=new_long, x="month", y="conc", hue="variable", split=True,palette=pal)
    # ax[1].legend(handles=ax.legend_.legendHandles, labels=["Observations", "Predictions"])
    ax[1].set_title("Violin Plot of Predictive Check")
    # ax.set_ylim(0,14)
    ax[1].grid()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/violin_train_size_" + str(config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Violin Plot": wandb.Image(im)})

    #FORECAST COMPONENTS
    m.plot_components(forecast)
    plt.show()

    eval_img = config["res_path"] + "/" + config["dependent"] + "/components_train_size_" + str(config["train_size"]) + '.png'
    fig.savefig(eval_img)
    wandb.save(eval_img)
    im = Image.open(eval_img)
    wandb.log({"Forecast Components": wandb.Image(im)})

    print("mean error: "+ str(me(actual , predicted)))
    print("mean average percentage error: ", str(mape(actual,predicted)))
    print("relative absolute error: ", str(rae(actual,predicted)))
    print("mean directional accuracy: ", str(mda(actual,predicted)))
    # print("mean average scaled error" , str(mase(actual,predicted,seasonality=360)))

    plt.close()
    plt.close()

    wandb.log({
        'mean_error': me(actual, predicted),
        'rel_abs_error': rae(actual, predicted),
        'mean_avg_per_error': mape(actual, predicted),
        'rt_sq_mean_error': rmse(actual, predicted)
        # 'mean_dir_acc': mda(actual, predicted)
        # 'mean_absolute_scaled_error': mase(actual, predicted, seasonality=360)
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="specify path to configuration file (yaml) ", type=str,
                        default="cfg/local_config.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb.login()
    run = wandb.init(mode=config["wandb_mode"],config = config)
    df = pd.read_csv(config["data_path"], low_memory=False)
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], format="%Y-%m-%d")  # required or else dates start at 1971! (WEIRD BUG HERE)
    df.loc[:,"month"] = df.date.dt.month
    df.loc[:,"doy_numeric"] = df.date.dt.dayofyear

    cap = config["cap"]
    if config["dependent"] == "diatomBiovol":
        df = df[df.diatomBiovol <= 1e7]

    # TWO DIMENSIONAL
    if config["temp"]:
        dfsubset = df.dropna(subset=[config["dependent"],config["predictor"]])

    if not config["temp"]:
        dfsubset = df.dropna(subset=[config["dependent"]])

    dfsubset.rename(columns={'date': 'ds', config["dependent"]: 'y', 'Beam_temperature_corrected':'temp' ,'AvgSolar': 'light'}, inplace=True)
    dfsubset["cap"] = cap

    # TRANSFORMATIONS
    if config["take_log"] and config["take_cube_root"]:
        print("ERROR: only 1 transformation!")

    if config["take_log"]:
        dfsubset.y = np.log(dfsubset.y)
    if config["take_cube_root"]:
        dfsubset.y = np.power(dfsubset.y,1/3)

    run = wandb.init(mode=config["wandb_mode"])
    main_sweep(config)