# utilities used for data cleaning

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler


class Cleaner:
    def __init__(self):
        pass

    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        drop columns
        """
        return df.drop(columns=columns)
    
    def drop_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        drop rows with nan values
        """
        return df.dropna()
    
    def drop_nan_column(self, df: pd.DataFrame, col:str) -> pd.DataFrame:
        """
        drop rows with nan values
        """
        return df.dropna(subset=[col])  
    
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """
        return df.drop_duplicates()
    
    def convert_to_datetime(self, df: pd.DataFrame, col: list) -> pd.DataFrame:
        """
        convert column to datetime
        """
        df[col] = df[col].apply(pd.to_datetime)
        return df
    
    def convert_to_string(self, df: pd.DataFrame, col = list) -> pd.DataFrame:
        """
        convert columns to string
        """
        df[col] = df[col].astype(str)
        return df
    
    def remove_whitespace_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        remove whitespace from columns
        """
        return df.columns.str.replace(' ', '_').str.lower()
       
    def percent_missing(self, df: pd.DataFrame) -> float:
        """
        calculate the percentage of missing values from dataframe
        """
        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMising = missingCount.sum()
        
        return round(totalMising / totalCells * 100, 2)
     
    def get_numerical_columns(self, df: pd.DataFrame) -> list:
        """
        get numerical columns
        """
        return df.select_dtypes(include=['float64']).columns.to_list()
    
    def get_categorical_columns(self, df: pd.DataFrame) -> list:    
        """
        get categorical columns
        """
        return  df.select_dtypes(include=['object','datetime64[ns]']).columns.to_list()
    
    def impute_zero(self, df: pd.DataFrame, column: list) -> pd.DataFrame:
        """
        imputes 0 inplace of NaN for a given columon(s)
        """
        df[column] = df[column].fillna(0)
        
        return df 

    def fill_missing_values_categorical(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        fill missing values with specified method
        """
        
        categorical_columns = df.select_dtypes(include=['object','datetime64[ns]']).columns
        
        if method == "ffill":
            
            for col in categorical_columns:
                df[col] = df[col].fillna(method='ffill')
                
            return df
        
        elif method == "bfill":
            
            for col in categorical_columns:
                df[col] = df[col].fillna(method='bfill')
                
            return df
        
        elif method == "mode":
            
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                
            return df
        else:
            print("Method unknown")
            return df
    
    def fill_missing_values_numeric(self, df: pd.DataFrame, method: str,columns: list = None) -> pd.DataFrame:
        """
        fill missing values with specified method
        """
        if(columns==None):
            numeric_columns = df.select_dtypes(include=['float64','int64']).columns
        else:
            numeric_columns=columns
        
        if method == "mean":
            for col in numeric_columns:
                df[col].fillna(df[col].mean(), inplace=True)
                
        elif method == "median":
            for col in numeric_columns:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            print("Method unknown")
        
        return df
    
    def normalizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        normalize numerical columns
        """
        norm = Normalizer()
        return pd.DataFrame(norm.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))
    
    def min_max_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """
        minmax_scaler = MinMaxScaler()
        return pd.DataFrame(minmax_scaler.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))
    
    def standard_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """
        standard_scaler = StandardScaler()
        return pd.DataFrame(standard_scaler.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))
    
    def detect_outliers(self, df:pd.DataFrame, threshold: int) -> list:
        """
        detect the indices of outliers using Z-method 
        """
        z_scores = df.apply(lambda x: np.abs((x - x.mean()) / x.std()))
        tr = threshold
        outliers = np.where(z_scores > tr)
        outlier_indices = [(df.index[i], df.columns[j]) for i, j in zip(*outliers)]
        return outlier_indices
        

    def handle_outliers(self, df:pd.DataFrame, indices:list, method:str) -> pd.DataFrame:
        """
        Handle Outliers of a specified column using the Z method
        """
        if method == 'mean':
            for idx, col_name in indices:
                column_mean = df[col_name].mean()
                df.iloc[idx, df.columns.get_loc(col_name)] = column_mean
        
        elif method == 'mode':
            for idx, col_name in indices:
                column_mode = df[col_name].mode()
                df.iloc[idx, df.columns.get_loc(col_name)] = column_mode

        elif method == 'median':
            for idx, col_name in indices:
                column_median = df[col_name].median()
                df.loc[idx, col_name] = column_median
        else:
            print("Method unknown")
    
        return df
    
    def find_agg(self,df:pd.DataFrame, agg_column:str, agg_metric:str, col_name:str, top:int, order=False )->pd.DataFrame:
        
        new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name).\
                            sort_values(by=col_name, ascending=order)[:top]
        
        return new_df

    def convert_bytes_to_megabytes(self,df, bytes_data):

        """
            This function takes the dataframe and the column which has the bytes values
            returns the megabytesof that value
            
            Args:
            -----
            df: dataframe
            bytes_data: column with bytes values
            
            Returns:
            --------
            A series
        """
        
        megabyte = 1*10e+5
        df[bytes_data] = df[bytes_data] / megabyte
        
        return df[bytes_data]
    
    def missing_values_table(self,df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # dtype of missing values
        mis_val_dtype = df.dtypes

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns