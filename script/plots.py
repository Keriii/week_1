import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plots:
    def plot_hist(self,df:pd.DataFrame, column:str, color:str)->None:
        plt.figure(figsize=(12, 7))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        sns.histplot(data=df, x=column, color=color, kde=True)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_count(self,df:pd.DataFrame, column:str) -> None:
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()
        
    def plot_bar(self,df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
        plt.figure(figsize=(12, 7))
        sns.barplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_heatmap(self,df:pd.DataFrame, title:str, cbar=False)->None:
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_box(self,df:pd.DataFrame, x_col:str, title:str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(self,df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.show()

    def plot_scatter(self,df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks( fontsize=14)
        plt.show()
