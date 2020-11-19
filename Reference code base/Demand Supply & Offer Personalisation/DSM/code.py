#@author - Ankur

import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='Demand Prediction XGBOOST models')
    parser.add_argument('--data-dir', type=str,default='/Users/ankur/Documents/Projects/Demand_Supply/Demand/data', help='folder for storing data')
    parser.add_argument('--save-dir', type=str,default='/Users/ankur/Documents/Projects/Demand_Supply/Demand/Deliverable' , help='folder for saving data')
    parser.add_argument('--data-frame', type=str,default= ['data_demand_city_date_nov10.csv'] , help='input dataset')
    parser.add_argument('--target-var', type=str,default= ['task_target'] , help='target variable')
    parser.add_argument('--date-var', type=str,default=  ['dt'] , help='idx features')
    parser.add_argument('--idx-var', type=str,default=  [] , help='idx features')
    parser.add_argument('--multi-discont', type=str,default=  [] , help='specify if there are multiclass discontinuous features')
    parser.add_argument('--text-var', type=str,default=  [] , help='text variables')
    parser.add_argument('--remove-var', type=str,default=  list(set(['task_target','demand_target']) - set(['task_target'])) , help='specify if we need to remove some features')
    print(args_string)
    args = parser.parse_args(args=args_string)
    return args
    
    
    
    
    import pandas as pd
import numpy as np
import os
import datetime
from datetime import time , date
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
from arguments import get_args
args = get_args()

class data_load():
    def __init__(self,dataframe , target , dtype_list , granularities , temporal_vars , time , to_embed , int_str ,univar_target_encodes  , bivar_target_encodes,multivar_target_encodes,  interactions ):
        self.dataframe = dataframe ; self.target = target ; self.dtype_list = dtype_list ; self.granularities = granularities ; self.temporal_vars = temporal_vars 
        self.time = time ; self.to_embed = to_embed ; self.int_str = int_str ; self.univar_target_encodes = univar_target_encodes ; 
        self.bivar_target_encodes = bivar_target_encodes ; self.multivar_target_encodes = multivar_target_encodes ; self.interactions = interactions
        self.dataframe = self.dataframe[(self.dataframe['drop_area'] != '0')] 
        #self.dataframe = self.dataframe[(self.dataframe['dt'] <  datetime.date(2019 ,9 ,30))]

        self._init_str_emb_dict_([self.dataframe ])  ; print('print the embed dict' , self.str_emb_dict , self.str_emb_len)
        self._init_str_replace_num_([self.dataframe ]) ; print(self.str_int)
        self.dataframe = self._init_temporal_daysince_([self.dataframe ]) ; print(self.dataframe.head(2) , self.temporal_daysince)
        self.dataframe = self._init_manual_temporal_([self.dataframe])
        #self.dataframe = self._init_temporal_vars_lag_([self.dataframe ]) ; print(self.dataframe.head(2) , self.temporal_lags) 
        self._init_count_dim_([self.dataframe]) ; print('embedding size' , self.emb_num_int , self.nonemb_num_int)
        self.dataframe = self._init_change_vars_([self.dataframe]) ; self.dataframe = self._init_one_hot_([self.dataframe])
        self._init_univar_target_encoding_([self.dataframe]) ; self._init_bivar_target_encoding_([self.dataframe]) ; self._init_multivar_target_encoding_([self.dataframe])
        print(self.univar_encoding) ; print(self.bivar_encoding) ; print(self.multivar_encoding)
        self.encoding = [] ; self.encoding += self.univar_encoding ; self.encoding += self.bivar_encoding; self.encoding += self.multivar_encoding 
        self.encoding = list(set(self.encoding) - set(['dt_day_Holiday_Flag_enc'])) + ['drop_areaHoliday_Flag_daysince_modified' , 'dt_weekday_weekend_flag'] 
        
        train_test , train , validation = self._init_sample_(self.dataframe) ; print(train.shape , validation.shape)
        self._init_encode_dict_([train]) 

        self.dataframe = self._init_enc_insert_([self.dataframe]) ;check = self.dataframe[(self.dataframe['drop_area'] != 'HSR,BDA')] #; check.to_csv('check.csv') ; print('check csv extracted')
        self._init_float_interactions_([self.dataframe])  
        train_test , train , validation = self._init_sample_(self.dataframe) 
        train , validation = self._init_rolling_target_(train_test) ; print(train.shape , validation.shape)
        self.xgb_features =[]  
        self.extra_features = ['offset' ]
        self.xgb_features += self.temporal_daysince ; self.xgb_features += self.str_int; self.xgb_features += self.one_hot_vars 
        self.xgb_features += self.encoded_feature_list; 
        self.xgb_features += self.interaction_feature_list 
        self.xgb_features += self.extra_features 
        self.xgb_features = np.unique(self.xgb_features)
        self._init_model_data_(train , validation)
        self.index_features = ['city','drop_area' , 'dt' ,'dt_week','dt_weekday','day_part' ,self.target[0]]
        self.index_feature_train = train[self.index_features] ; self.index_feature_validation = validation[self.index_features] 


    #Create embedding dictionary for all string features and insert them into data

    def _init_str_emb_dict_(self,dataframe):
        str_emb_dict = {} ; str_emb_len = {}
        for df in dataframe:
            for key in df.keys():
                if key in self.dtype_list[1]:
                    df[key + '_emb'] = df[key]
                    str_emb_dict[key + '_emb'] = np.unique(df[key + '_emb'])
                    str_emb_len[key + '_emb'] = len(np.unique(df[key + '_emb']))
        self.str_emb_dict = str_emb_dict ; self.str_emb_len = str_emb_len

    def _init_str_replace_num_(self, dataframe):
        self.str_int = []
        for df in dataframe:
            for key in self.str_emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.str_emb_dict.get(key)) , list(range(len(self.str_emb_dict.get(key)))))
                    self.str_int += [key]


    #Create all the temporal features

    def _init_temporal_daysince_(self,dataframe):
        self.temporal_daysince = []
        for df in dataframe:
            for gran_list in self.granularities:
                print('granlist' , gran_list , gran_list[0] , gran_list[1])
                df = df.sort_values(gran_list[1] , ascending = True) ; print('sorted' , df.head(10))
                df[gran_list[0][0] + '_seq'] = df.groupby(gran_list[0])[self.time].cumcount() + 1
                #print('datatype of date' , df['Date'].dtype)
                df[gran_list[0][0] + '_gapsince'] = df.groupby(gran_list[0])[self.time].diff()
                print('check the data' , df.head(2))
                for key in self.temporal_vars:
                    print('temporal_daysince key' , key)
                    df[gran_list[0][0] + key + '_daysince'] = df[gran_list[0]].groupby((df[key] != df[key].shift()).cumsum()).cumcount() + 1
                    df[gran_list[0][0] + key + '_daysince'] = np.where(df[key]  != 0, 0, df[gran_list[0][0] + key + '_daysince'])
                    self.temporal_daysince += [gran_list[0][0] + '_seq' , gran_list[0][0] + key + '_daysince']
        return df

    def _init_manual_temporal_(self, dataframe):
        self.holiday_encode = []
        for df in dataframe:
            for key in df.keys():
                if key in list(set(self.temporal_daysince) - set(['drop_area_seq'])):
                    df[key + '_modified'] = np.where(df[key] > 5 , 100 , df[key])
                    self.holiday_encode += [key + '_modified'] 
                if key in ['dt_weekday']:
                    df[key + '_weekend_flag'] = np.where(df[key] >= 5 , 1 , 0)
        return df

    #One hot encoding for the required features
        
    def _init_change_vars_(self,dataframes):
        self.one_hot = []
        for df in dataframes:
            for key in df.keys():
                if key in self.int_str:
                    df['str_key'] = key
                    df[key+'_string'] = df[key]
                    df[key+'_string'] = df['str_key'].astype(str).str.cat(df[key+'_string'].astype(str))
                    self.one_hot += [key+'_string']
                    print('self.one_hot' , self.one_hot)
        return df

    def _init_one_hot_(self,dataframes):
        one_hot_vars = []
        for df in dataframes:
            for key in self.one_hot:
                print(key)
                one_hot = pd.get_dummies(df[key])
                df = df.join(one_hot)
                print('check the shape' , df.shape , df.head(2))
                one_hot_vars  += list(one_hot.columns.values) ; self.one_hot_vars = one_hot_vars ; print('self.one_hot_vars' , self.one_hot_vars)
        return df

    #univariate target encoding 
    def _init_univar_target_encoding_(self,dataframes):
        univar_encoding = [] 
        for df in dataframes:
            for key in self.univar_target_encodes:
                print('univar_yupp' , key)
                df[key+'_enc'] = df[key]			
                univar_encoding += [key+'_enc']
        self.univar_encoding = (univar_encoding)

    #bivariate target encoding 
    def _init_bivar_target_encoding_(self,dataframes):
        bivar_encoding = [] 
        for df in dataframes:
            for i , key1 in enumerate(self.bivar_target_encodes):
                if (i+1 < len(self.bivar_target_encodes)):				
                    for j , key2 in enumerate(self.bivar_target_encodes[i+1:]):
                        if (j+1 < len(self.bivar_target_encodes[i+1:])):
                            print('bivar_yupp' , key1, key2)
                            df[key1 +'_' + key2 + '_enc'] = df[key1].astype(str).str.cat(df[key2].astype(str))
                            df['city_quarter_enc'] = df['city'].astype(str).str.cat(df['dt_quarter'].astype(str))
                            df['area_quarter_enc'] = df['drop_area'].astype(str).str.cat(df['dt_quarter'].astype(str))
                            bivar_encoding += [key1 +'_' + key2 + '_enc', 'area_quarter_enc', 'area_quarter_enc']
        self.bivar_encoding = (bivar_encoding)

    #multivar target encoding 
    def _init_multivar_target_encoding_(self,dataframes):
        multivar_encoding = [] ; self.encoding = []
        for df in dataframes:
            for i , key1 in enumerate(self.multivar_target_encodes):
                if (i+1 < len(self.multivar_target_encodes)):					
                    for j , key2 in enumerate(self.multivar_target_encodes[i+1:]):
                        if (j+1 < len(self.multivar_target_encodes[i+1:])): 
                            for key3 in self.multivar_target_encodes[i+1:][j+1:]:
                                print('yupppppp' , key1 , key2 , key3)
                                df[key1+'_enc'] = df[key1]
                                df[key1 +'_' + key2 + '_enc'] = df[key1].astype(str).str.cat(df[key2].astype(str))
                                df[key1 + '_' + key2 +'_' + key3 + '_enc'] = df[key1].astype(str) + df[key2].astype(str).str.cat(df[key3].astype(str))
                                #df['area_quarter_weekday_enc'] = df['drop_area'].astype(str) + df['dt_quarter'].astype(str).str.cat(df['dt_weekday'].astype(str))
                                multivar_encoding += [ key1+'_enc', key1 +'_' + key2 + '_enc', key1 + '_' + key2 +'_' + key3 + '_enc', 'area_quarter_weekday_enc']
        self.multivar_encoding = (multivar_encoding)

    def _init_encode_dict_(self,dataframes):
        encoded_feature_list = []
        emb_dict1 = {}
        encode_dict = {}
        for df in dataframes:
            for key in df.keys():
                if key in self.encoding:
                    emb_dict1[key] = np.unique(df[key]) ; print('emb_dict1', key ,emb_dict1[key].shape)
                    encode = df.groupby([key])[self.target].mean().reset_index()
                    encode_dict[key] = encode[self.target].values
                    encoded_feature_list += [key]
        self.emb_dict1 = emb_dict1 ; self.encode_dict = encode_dict
        self.encoded_feature_list = encoded_feature_list 
        #print( 'emb_dict1' ,self.emb_dict1) ; print( 'encode_dict' ,self.encode_dict)

    def _init_enc_insert_(self,dataframes):
        self.interactions = []
        for df in dataframes:
            for key in self.emb_dict1.keys():
                if key in df.keys():
                    print( 'Entered Encode insert : lets see where ?',key )
                    df[key] = df[key].replace(list(self.emb_dict1.get(key)) , list(self.encode_dict.get(key)))
                    self.interactions += [key]
            return df

    #Interactions and Normalization of the features
    def _init_float_interactions_(self,dataframes):
        interaction_feature_list = []
        print('Entered interactions')
        for df in dataframes:
            for i, key1 in enumerate(self.interactions):
                if(i+1 < len(self.interactions)):
                    for key2 in self.interactions[i+1:]:
                        df[key1 + '_' + key2 + '_interaction'] = df[key1]/df[key2]
                        df[key1 + '_' + key2 + '_interaction'].replace(np.inf, 0, inplace=True)
                        df[key1 + '_normalized'] = df[key1]/np.mean(df[key1])
                        df[key1 + '_' + key2 + '_norm_mult'] = (df[key1]/np.mean(df[key1]))*(df[key2]/np.mean(df[key2]))
                        interaction_feature_list += [key1 + '_' + key2 + '_interaction', key1 + '_normalized', key1 + '_' + key2 + '_norm_mult']
        self.interaction_feature_list = interaction_feature_list

    def _init_count_dim_(self, dataframe):
        self.emb_num_int , self.nonemb_num_int = 0,0
        for df in dataframe:
            for key in df.keys():
                if key in self.to_embed:
                    self.emb_num_int += 1
                else:
                    self.nonemb_num_int += 1
    

    def _init_sample_(self,data):
        train = data[(data['dt'] <  datetime.date(2019 , 10 ,7) )] ; train['train_flag'] = 1
        validation = data[(data['dt'] <  datetime.date(2019 ,12 ,30) ) & (data['dt'] >=  datetime.date(2019 , 10 ,7) )] ;  validation['train_flag'] = 0
        train_test = train.append(validation, ignore_index=True)

        # train = data[(data['dt'] <  datetime.date(2019 , 10 ,14) )] ; train['train_flag'] = 1
        # validation = data[(data['dt'] <  datetime.date(2019 ,11 ,4) ) & (data['dt'] >=  datetime.date(2019 ,10 ,14) )] ;  validation['train_flag'] = 0
        # train_test = train.append(validation, ignore_index=True)

        # train = data[(data['dt'] <  datetime.date(2019 , 8 ,5) )] ; train['train_flag'] = 1
        # validation = data[(data['dt'] <  datetime.date(2019 ,9 ,30) ) & (data['dt'] >=  datetime.date(2019 ,8 ,5) )] ;  validation['train_flag'] = 0
        # train_test = train.append(validation, ignore_index=True)
        return train_test , train , validation
    
    def _init_polynomial_reg_(x,p1,p0):
        return p1*(x) + p0

    def _init_rolling_target_(self,train_test):
        #data = train[(train['quantity'] >= 0)]
        data = train_test
        data['rank_day_desc'] = data.groupby(['day_part','drop_area'])['dt'].rank(ascending=False).astype(int)
        data['rank_day_asc'] = data.groupby(['day_part','drop_area'])['dt'].rank(ascending=True).astype(int)
        data = data.sort_values(['day_part','drop_area','dt'],ascending=True)

        train = data[(data['train_flag'] == 1)] ; validation = data[(data['train_flag'] == 0)] ; print('shaped ins offset1',train.shape , validation.shape)
        train['roll_target1'] = train.groupby(['day_part','drop_area'])[self.target[0]].shift(1).rolling(28).mean()
        train['roll_target2'] = train.groupby(['day_part','drop_area'])[self.target[0]].shift(1).rolling(21).mean()
        train['roll_target3'] = train.groupby(['day_part','drop_area'])[self.target[0]].shift(1).rolling(14).mean()
        train['roll_target4'] = train.groupby(['day_part','drop_area'])[self.target[0]].shift(1).rolling(7).mean()
        train['offset'] = np.nan
        train['offset'] = train.offset.fillna(train.roll_target1).fillna(train.roll_target2).fillna(train.roll_target3).fillna(train.roll_target4)
        data = data[(data['rank_day_asc'] > 7)]
        train['week_sum'] = train.groupby(['drop_area' , 'dt_week'])[self.target[0]].transform(np.mean) 
        train['month_sum'] = train.groupby(['drop_area' , 'dt_month'])[self.target[0]].transform(np.mean)

        train1 = train[(train['dt_week'] > 21)]

        validation_base = train1[(train1['drop_area_emb'] == 999)] ; validation_base = validation_base[['drop_area_emb' , 'rank_day_asc' , 'offset']]
        for area in list(train1['drop_area_emb'].unique()):
            print('area' , area)
            train_df = train1[(train1['drop_area_emb'] == area)]  ; print('train_df1' , train_df.shape)
            train_offset_mean = train_df[(train_df['dt_week'] == max(train_df['dt_week']) )]
            train_offset_mean = np.mean(train_offset_mean['offset']) 
            validation_df = validation[(validation['drop_area_emb'] == area)] ; validation_df['offset'] = 0
            train_df = train_df[['rank_day_asc' , 'offset']]  ; print('train_df2' , train_df.shape , train_df.head(10) )
            validation_df = validation_df[['drop_area_emb' , 'rank_day_asc' , 'offset']]  ; validation_pred = validation_df['rank_day_asc']
            x = train_df['rank_day_asc'].values ; y = train_df['offset'].values ; print('ab ',x, y)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            print('params' ,slope, intercept )
            validation_df['offset'] = slope * validation_df['rank_day_asc'] + intercept
            validation_df['offset'] = np.where(validation_df['offset'] < 5  ,train_offset_mean , validation_df['offset'] )
            print('validation_df' , validation_df.head(2))
            validation_base = validation_base.append(validation_df ,  ignore_index=True) ; print('validation_base',validation_base.head(2) ,validation_base.shape)
        
        print('check the val shapes' , validation.shape , validation_base.shape)
        validation = pd.merge(validation , validation_base , on = ['drop_area_emb' , 'rank_day_asc'] , how = 'left')
        
        print('check the val shapes' , validation.shape , validation_base.shape)
        validation = pd.merge(validation , validation_base , on = ['drop_area_emb' , 'dt_month'] , how = 'left')

        # validation_base = train1[(train1['drop_area_emb'] == 999)] ; validation_base = validation_base[['drop_area_emb' , 'dt_month' , 'month_sum']]
        # for area in list(train1['drop_area_emb'].unique()):
        #     print('area' , area)
        #     train_df = train1[(train1['drop_area_emb'] == area)]  ; print('train_df1' , train_df.shape)
        #     train_offset_mean = train_df[(train_df['dt_month'] == max(train_df['dt_month'])  )]
        #     train_offset_mean = np.mean(train_offset_mean['month_sum']) 
        #     validation_df = validation[(validation['drop_area_emb'] == area)] ; validation_df['month_sum'] = 0
        #     train_df = train_df[['dt_month' , 'month_sum']]  ; print('train_df2' , train_df.shape , train_df.head(10) )
        #     validation_df = validation_df[['drop_area_emb' , 'dt_month' , 'month_sum']]  ; validation_pred = validation_df['dt_month']
        #     x = train_df['dt_month'].values ; y = train_df['month_sum'].values ; print('ab ',x, y)
        #     slope, intercept, r_value, p_value, std_err = linregress(x, y)
        #     print('params' ,slope, intercept )
        #     validation_df['month_sum'] = slope * validation_df['dt_month'] + intercept
        #     validation_df['month_sum'] = np.where(validation_df['month_sum'] <1 ,train_offset_mean , validation_df['month_sum'] )
        #     print('validation_df' , validation_df.head(2))
        #     validation_base = validation_base.append(validation_df ,  ignore_index=True) ; validation_base.drop_duplicates(inplace=True)
        #     print('validation_base',validation_base.head(2) ,validation_base.shape)
        
        # print('check the val shapes' , validation.shape , validation_base.shape)
        # validation = pd.merge(validation , validation_base , on = ['drop_area_emb' , 'dt_month'] , how = 'left')

        #extract data for checks
        check = data[(data['drop_area'] == 'HSR,BDA')] ; check.to_csv('check.csv') ; print('check csv extracted')
        return train , validation

    def _init_model_data_(self, train , validation):
        self.target_train = train[self.target] ; self.target_val = validation[self.target]
        train = train.reindex(columns=self.xgb_features) ; validation = validation.reindex(columns=self.xgb_features)
        train = train[self.xgb_features] ; validation = validation[self.xgb_features]
        print('lets have a look at the model shapes',train.shape , validation.shape , train.columns)
        train = train.as_matrix() ; validation = validation.as_matrix()
        self.xg_train = xgb.DMatrix(train, label=self.target_train,feature_names=self.xgb_features)
        self.xg_val = xgb.DMatrix(validation, label=self.target_val,feature_names=self.xgb_features)








#@author - Ankur

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from arguments import get_args
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import f1_score

args = get_args()

class xgb_model():
    def __init__(self,xg_train ,xg_val,target_train ,target_val,index_feature_train ,index_feature_validation ,nrounds = 500):
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'reg:squarederror'
        param["eval_metric"] = 'rmse'
        param['eta'] = 0.025
        param['max_depth'] = 8
        param['min_child_weight']=1
        param['subsample']= 0.75
        param['colsample_bytree']=0.75
        #param['silent'] = 0
        param['verbose'] = 1
        self.param = param
        self.nrounds = nrounds
        print('params' , self.param)
        self._init_model_training_iter_(xg_train,xg_val,target_train,target_val , index_feature_train ,index_feature_validation)

    def _init_model_training_iter_(self,xg_train,xg_val,target_train,target_val , index_feature_train ,index_feature_validation):
        print('start the training')
        self.model = xgb.train(self.param, xg_train, self.nrounds)
        print('training complete' , self.model)
        self.pred_train = self.model.predict(xg_train) ; self.pred_val = self.model.predict(xg_val) 
        self.target_train = target_train.values ; self.target_val = target_val.values
        submit_granular = pd.DataFrame(self.pred_val) ; submit_granular.columns = ['pred_val']
        submit_granular = pd.concat([index_feature_validation , submit_granular] , axis = 1)
        print('target length' ,  len(self.target_train) , len(self.target_val))
        submit_granular['MAPE'] = 100 - (abs(submit_granular[args.target_var[0]] - submit_granular['pred_val'])/submit_granular[args.target_var[0]])* 100
        submit_interim = submit_granular[(submit_granular['day_part'] >0 )]
        score = np.mean(submit_interim['MAPE'])
        submit_overall = submit_interim.groupby(['drop_area']).agg({args.target_var[0] : 'sum' , 'pred_val' : 'sum' , 'MAPE' : 'mean'}).reset_index()
        submit_overall.columns = ['drop_area' , 'Actual_Sum' , 'Predicted_Sum' , 'MAPE_AVG']
        submit_overall['PE'] = 100 - (abs(submit_overall['Actual_Sum'] - submit_overall['Predicted_Sum'])/submit_overall['Actual_Sum'])* 100
        submit_granular.to_csv(os.path.join(args.save_dir , 'submit_granular_Demand_oct21_tasks.csv') , index = False) 
        submit_overall.to_csv(os.path.join(args.save_dir , 'submit_overall_Demand_oct21_tasks.csv') , index = False) 
        print('model results' , score)
        self._init_feature_imp_(self.model)

    def _init_feature_imp_(self,model):
        feature_imp = self.model.get_score(importance_type='gain')
        feature_imp = list(feature_imp.items()) ; feature_imp = pd.DataFrame(feature_imp , columns = ['Feature' , 'Gains'])
        feature_imp = feature_imp.sort_values(by = ['Gains'] , ascending = False)
        feature_imp.to_csv(os.path.join(args.save_dir , 'feature_imp_Demand_oct21_tasks.csv'), index=False)
        print('feature_imp' , feature_imp)





#@author - Ankur

import pandas as pd
import numpy as np
import os
from reader import data_read
from loader import data_load
from model import xgb_model
from arguments import get_args


if __name__ == '__main__':
    print('Yo! Demand Models')
    args = get_args()
    print(args)

    #Call the class data_read()
    data = data_read(args.data_dir, args.data_frame ,args.date_var , args.target_var , args.idx_var , args.multi_discont , args.text_var , args.remove_var )

    #call the class data_load
    dataframe = data.data 
    target = args.target_var 
    dtype_list = data.dtype_list  
    granularities = [[['drop_area'] , ['drop_area' , 'dt']] , [['drop_area'] , ['drop_area' , 'dt' , 'day_part']]]  
    temporal_vars = ['Holiday_Flag' , 'Holiday_Type']
    time = args.date_var
    to_embed = []
    int_str = ['dt_month', 'dt_weekday' ,'dt_quarter' ,'Holiday_Type' ,'city','drop_areaHoliday_Flag_daysince_modified']
    univar_target_encodes = ['city','drop_area', 'dt_weekday','dt_quarter' ,'drop_areaHoliday_Flag_daysince_modified','dt_weekday_weekend_flag']
    #univar_target_encodes = []
    bivar_target_encodes = ['city','drop_area','drop_areaHoliday_Flag_daysince_modified','dt_weekday_weekend_flag','city']
    #bivar_target_encodes = []
    multivar_target_encodes = ['city','drop_area','dt_weekday']  
    #multivar_target_encodes = []
    interactions =  []

    load = data_load(dataframe , target , dtype_list , granularities , temporal_vars , time, to_embed , int_str, univar_target_encodes  , bivar_target_encodes,multivar_target_encodes,  interactions )

    #call the class model
    model = xgb_model(load.xg_train ,load.xg_val,load.target_train ,load.target_val,load.index_feature_train ,load.index_feature_validation )







#@author - Ankur

import pandas as pd
import numpy as np
import os
import datetime
from dask import dataframe as dd 
from arguments import get_args

class data_read():
    def __init__(self,dir ,dataframe_list  ,date_list , target_list ,idx, multiclass_discontinuous , text, remove_list):
        self.dir = dir ; self.date_list = date_list ; self.target_list = target_list ;self.idx = idx
        self.multiclass_discontinuous = multiclass_discontinuous ; self.text = text ;self.remove_list = remove_list
        self.df_list = self._init_read_(dataframe_list)
        self.data = self._init_dtype_(self.df_list)
        self._init_number_dtype_split_([self.data])
        self._init_impute_vars_([self.data])
        self._init_number_dtype_split_([self.data])
        self.dtype_list = [self.date_list,self.strings,self.number,self.number_binary, self.number_multi ,self.number_continuous, self.multiclass_discontinuous ,self.text ,self.target_list]
        self.dtype_list_name = ['date_list', 'strings', 'number','number_binary', 'number_multi' ,'number_continuous','multiclass_discontinuous' ,'text' ,'target_list']
        self._init_data_check_(self.data , self.dtype_list , self.dtype_list_name)

    def _init_read_(self,dataframes):
        df_list = []
        for df in dataframes:
            data = dd.read_csv(os.path.join(self.dir ,df) , na_values = ' ', low_memory=False , assume_missing = True); data = data.compute()
            print(data.columns , data.head(5) ,'the shape is here' , data.shape) 
            df_list.append(data)
        return df_list

    def _init_dtype_(self,dataframes):
        strings = [] ; number = []
        for df in dataframes:
            strs = list(df.select_dtypes(include = [np.object])) ; strings += strs
            nbr = list(df.select_dtypes(exclude = [np.object])) ; number += nbr
            for key in df.keys():
                if key in self.date_list:
                    df[key] = pd.to_datetime(df[key])
                    df[key + '_date'] = df[key].dt.date
                    df[key + '_year'] = df[key].dt.year
                    df[key + '_month'] = df[key].dt.month
                    df[key + '_quarter'] = df[key].dt.quarter
                    df[key + '_week'] = df[key].dt.week
                    df[key + '_day'] = df[key].dt.day
                    df[key + '_weekday'] = df[key].dt.weekday
                    df = df.sort_values([key],ascending=True)
                    print('check the df' , df.head(2))
                    number += [key + '_year' , key + '_month', key + '_quarter' ,key + '_week', key + '_day', key + '_weekday' ]
            self.strings = list(set(np.unique(strings)) - set((self.date_list + self.idx + self.multiclass_discontinuous + self.text + self.target_list +self.remove_list)))
            self.number = list(set(np.unique(number)) - set((self.idx + self.remove_list + self.multiclass_discontinuous +self.target_list)))
            return df

    def _init_number_dtype_split_(self,dataframes):
        number_binary = [] ; number_multi = [] ; number_continuous = []
        for df in dataframes:
            for key in df.keys():
                if (key in self.number and len(np.unique(df[key])) <=2):
                    number_binary += [key]
                if (key in self.number and (len(np.unique(df[key])) > 2 and len(np.unique(df[key])) <= 12)):
                    number_multi += [key]
                if (key in self.number and len(np.unique(df[key])) > 12 ):
                    number_continuous += [key]
            self.number_binary = np.unique(number_binary) ; self.number_multi = np.unique(number_multi) ;self.number_continuous = np.unique(number_continuous)

    def _init_impute_vars_(self,dataframes):
        for df in dataframes:
            for key in df.keys():
                if key in self.strings:
                    df[key].fillna('Impute',inplace = True)
                if key in self.number_binary:
                    df[key].fillna(0,inplace = True)
                if key in self.number_multi:
                    df[key].fillna(0,inplace = True)
                if key in self.number_continuous:
                    df[key].fillna(0 , inplace = True)

    def _init_data_check_(self , df_list , dtype_list , dtype_list_name):
        for i in range(0,len(self.dtype_list)):
            print('Data types' , i , self.dtype_list_name[i], self.dtype_list[i])
