#@author - Ankur

import argparse
import warnings
warnings.filterwarnings("ignore")

def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='Customer Category Propensity model')
    parser.add_argument('--data-dir', type=str,default='/Users/ankur/Documents/Projects/offer_personalisation/offer_v2/data/20191120', help='folder for storing data')
    parser.add_argument('--save-dir', type=str,default='/Users/ankur/Documents/Projects/offer_personalisation/offer_v2/experiments/data/20191120' , help='folder for saving data')
    parser.add_argument('--modelsave-dir', type=str,default='/Users/ankur/Documents/Projects/offer_personalisation/offer_v2/model/20191120' , help='folder for saving models')
    parser.add_argument('--data-frame', type=str,default= ['data.csv'] , help='input dataset')
    parser.add_argument('--target-var', type=str,default= [] , help='target variable')
    parser.add_argument('--date-var', type=str,default=  ['task_created_on'] , help='idx features')
    parser.add_argument('--idx-var', type=str,default=  [] , help='idx features')
    parser.add_argument('--multi-discont', type=str,default=  [] , help='specify if there are multiclass discontinuous features')
    parser.add_argument('--text-var', type=str,default=  [] , help='text variables')
    parser.add_argument('--remove-var', type=str,default=  list(set([]) - set([])) , help='specify if we need to remove some features')
    print(args_string)
    args = parser.parse_args(args=args_string)
    return args
    
    
    
    #@author - Ankur

import pandas as pd
import numpy as np
import os
import datetime
import xgboost as xgb
from datetime import time , date
from arguments import get_args
import warnings
warnings.filterwarnings("ignore")
args = get_args()

class data_load():
	def __init__(self,dataframe , number_binary , number_multi , number_continuous):
		self.dataframe = dataframe ; self.number_binary = number_binary ; self.number_multi = number_multi ; self.number_continuous = number_continuous 
		
		#Train Test timeframes
		#Train date => 22 April 2019 to 20 Oct 2019 ; target train => 21 Oct 2019 to 3 Nov 2019
		#Test date =>  6 May 2019 to 3 Nov 2019 ; target test future => 4 Nov 2019 to 17 Nov 2019
		self.train_st_dt = datetime.date(2019 ,4 ,22) ;self.train_ed_dt = datetime.date(2019 ,10 ,20); self.train_tar_st_dt = datetime.date(2019 ,10,21) ; self.train_tar_ed_dt = datetime.date(2019 ,11 ,3)
		self.val_st_dt = datetime.date(2019 ,5 ,6) ;self.val_ed_dt = datetime.date(2019 ,11 ,3) ; self.val_tar_st_dt = datetime.date(2019 ,11 ,4); self.val_tar_ed_dt = datetime.date(2019 ,11 ,17)
		
		self.date_col  = ['task_created_on_date']  ; self.user_id = ['user_id'] ; self.category = ['task_category']
		self.dt_index = ['user_id' , 'user_cat_id']
		self.dt_transpose = ['task_created_on_weekday' , 'task_created_on_month',	'task_created_on_week',	'task_created_on_day', 'city_id','task_category']
		self.exception_number_features = ['task_created_on_year' , 'city_id' ,'task_created_on_month' , 'task_created_on_weekday' , 'parent_task_id' ,'runner_id' ,'task_created_on_day', 'task_created_on_week', 'task_id' , 'user_id'   ]
		self.index_features = ['user_id',	'task_category'	,'user_cat_id',	'dummy' , 'target']
		self.target = 'target' ; print('passed all assignments')

		train_data , train_target_data , train_unique_data,val_data , val_target_data , val_unique_data = self._init_impute_train_val_([self.dataframe])
		
		print('shapes of data' , train_data.shape , train_target_data.shape , val_data.shape , val_target_data.shape )

		df_list_daysince= self._init_ordergap_levels_([train_data , val_data]) ; print('lets get shape _init_days_since_',len(df_list_daysince) ,df_list_daysince[0].shape ,df_list_daysince[1].shape ,df_list_daysince[2].shape ,df_list_daysince[3].shape )
		df_list_dt_dist = self._init_dt_dist_([train_data , val_data]) ; print('lets get shape _init_dt_dist_',len(df_list_dt_dist) ,df_list_dt_dist[0].shape ,df_list_dt_dist[1].shape ,df_list_dt_dist[2].shape ,df_list_dt_dist[3].shape )
		train_burn , val_burn = self._init_burn_dist_([train_data , val_data])
		df_list_base_feature = self._init_base_feature_([train_data , val_data]) ; print('lets get shape _init_base_feature_',len(df_list_base_feature) ,df_list_base_feature[0].shape ,df_list_base_feature[1].shape ,df_list_base_feature[2].shape ,df_list_base_feature[3].shape )

		#create dataframe lists for base data, target and feature
		self.base_data_list = [] ; self.base_data_list.append(train_unique_data) ;  self.base_data_list.append(val_unique_data)
		self.target_data_list = [] ; self.target_data_list.append(train_target_data) ;  self.target_data_list.append(val_target_data)
		self.feature_data_list = [] ; self.feature_data_list.append(df_list_daysince) ;  self.feature_data_list.append(df_list_dt_dist) ; self.feature_data_list.append(df_list_base_feature)
		self.burn_data_list = [] ; self.burn_data_list.append(train_burn) ; self.burn_data_list.append(val_burn)


		train , validation = self._init_train_val_consolidation_(self.base_data_list , self.target_data_list , self.burn_data_list)  
		print('yo! here starts the consolidation',train.head(2) , train.shape , validation.head(2) , validation.shape )

		#Clean the data for modelling
		df_list_treated = self._init_train_val_treat([train , validation])  ;  print('lets get shape df_list_treated',len(df_list_treated) ,df_list_treated[0].shape ,df_list_treated[1].shape)
		train , validation = df_list_treated[0] , df_list_treated[1]
		self.features = list(set(train.columns).intersection(validation.columns)) 
		self.features = list(set(self.features) - set(self.index_features))
		self.index_feature_train = train[self.index_features] ; self.index_feature_validation = validation[self.index_features] ; 
		self.sim_features = self.index_features  + self.features 
		train.to_csv(os.path.join(args.save_dir , 'check.csv') , index = False) 
		#check = train[(train['user_id'] == 83)]  ; check.to_csv('check.csv') ; print('shape' ,check.shape , self.features )
		#train.to_csv('train.csv' , index = False) ; validation.to_csv('validation.csv' , index = False)

		# # #=========================Optional==========================================
		# # #After creating the train and validation data
		# # train = dd.read_csv('train.csv', na_values = ' ', low_memory=False , assume_missing = True) ; train = train.compute()
		# # validation = dd.read_csv('validation.csv', na_values = ' ', low_memory=False , assume_missing = True) ; validation = validation.compute()
		# # #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		#create the model data
		xg_train , xg_val = self._init_model_data_(train , validation)
		print('model data created')

	#create train and validation set along with respective targets
	def _init_impute_train_val_(self,dataframe):
		for df in dataframe:
			for key in df.keys():
				#print('entered key' , key)
				if key in self.date_col:
					print('entered _init_impute_train_val_' , key)
					df['user_cat_id'] = df['user_id'].astype(str).str.cat(df['task_category'].astype(str))
					df['promotion_id'] = np.where(df['promotion_id'] > 0 ,1 , 0) ; df['offer_id'] = np.where(df['offer_id'] > 0 ,1 , 0) ; df['runner_task_status'] = np.where(df['runner_task_status'] == 'COMPLETED' ,1 , 0) 
					train_data = df[(df[key] >= self.train_st_dt) & (df[key] <= self.train_ed_dt)] ; train_unique_data = train_data.groupby(['user_id','task_category' ,'user_cat_id'])['task_id'].size().reset_index() ;train_unique_data.columns = ['user_id','task_category' ,'user_cat_id', 'dummy'] ; train_unique_data['dummy'] = 1 ;print(train_unique_data.head(2))
					train_target_data = df[(df[key] >= self.train_tar_st_dt) & (df[key] <= self.train_tar_ed_dt)] ; train_target_data = train_target_data.groupby(['user_cat_id'])['task_id'].size().reset_index() ;train_target_data.columns = ['user_cat_id' , 'target'] ; train_target_data['target'] = 1 ;print(train_target_data.head(2))
					val_data = df[(df[key] >= self.val_st_dt) & (df[key] <= self.val_ed_dt)]  ; val_unique_data = val_data.groupby(['user_id','task_category' ,'user_cat_id'])['task_id'].size().reset_index() ;val_unique_data.columns = ['user_id','task_category' ,'user_cat_id', 'dummy'] ; val_unique_data['dummy'] = 1 
					val_target_data = df[(df[key] >= self.val_tar_st_dt) & (df[key] <= self.val_tar_ed_dt)] ; val_target_data = val_target_data.groupby(['user_cat_id'])['task_id'].size().reset_index() ; val_target_data.columns = ['user_cat_id' , 'target'] ; val_target_data['target'] = 1 
		return train_data , train_target_data , train_unique_data, val_data , val_target_data , val_unique_data

	def _init_ordergap_levels_(self,dataframe):
		df_list = [] ; self.order_gap_features = []
		for df in dataframe:
			print('enter df', df.shape)
			for key in self.dt_index:
				print('which key?' , key)
				df['start_dt'] = np.min(df['task_created_on_date'])  ; df['end_dt'] = np.max(df['task_created_on_date']) 
				df[key + '_seq'] = df.groupby(key)[self.date_col].cumcount() + 1
				df[key + '_seq' + '_rank'] = df.groupby(key)[key + '_seq'].rank(ascending=False , method='dense').astype(int)
				df[key + '_first'] = df.groupby(key)['task_created_on_date'].transform(np.min)
				for i in [1,2,3]:
					df[key + '_prev' + str(i)] = df.groupby([key])['task_created_on_date'].shift(i)

				df[key + '_aog'] = (df['task_created_on_date']  - df[key + '_prev1']).dt.days
				df[key + '_first'] = (df['end_dt']  - df[key + '_first']).dt.days
				df[key + '_prev'] = (df['end_dt']  - df['task_created_on_date']).dt.days
				df[key + '_prev1'] = (df['end_dt']  - df[key + '_prev1']).dt.days
				df[key + '_prev2'] = (df['end_dt']  - df[key + '_prev2']).dt.days
				df[key + '_prev3'] = (df['end_dt']  - df[key + '_prev3']).dt.days
				df[key + '_aog'] = df.groupby(key)[key + '_aog'].transform(np.mean)
				df_key = df[(df[key + '_seq' + '_rank'] == 1 )]
				df_key = df_key[[key , key + '_seq' , key + '_seq' + '_rank' ,key + '_first' , key + '_prev' , key + '_prev1' , key + '_prev2',
				key + '_prev3' , key + '_aog' ]]
				self.order_gap_features += [ key + '_seq' , key + '_seq' + '_rank' ,key + '_first' , key + '_prev' , key + '_prev1' , key + '_prev2',
				key + '_prev3' , key + '_aog' ]
				print('shape of df_key' , df_key.shape)
				df_list.append(df_key)
		return df_list
	
	def _init_dt_dist_(self,dataframe):
		df_list = []
		for df in dataframe:
			for key_level in self.dt_index:
				#print('key_level',key_level)
				base_data = pd.DataFrame(df[key_level].unique()) ; base_data.columns = [key_level]
				for key_dt in self.dt_transpose: 
					#print('key_dt',key_dt)
					df['str_key'] = key_dt ; df['str_key1'] = key_level ; df['str_key'] = df['str_key'].astype(str).str.cat(df['str_key1'].astype(str)) ; df[key_dt + '_str'] = df['str_key'].astype(str).str.cat(df[key_dt].astype(str))
					data = df.groupby([key_level , key_dt + '_str'])[key_dt + '_str'].agg({'task_id' : 'size'}).reset_index()
					interm_data = df.groupby([key_level])[key_dt + '_str'].agg({'task_created_on_date' : 'size'}).reset_index()
					data = pd.merge(data, interm_data, on = key_level , how= 'left')
					data['trans_percent_dist'] = data['task_id']/data['task_created_on_date'] ; data = data[[key_level ,key_dt + '_str' ,'trans_percent_dist']]
					#print('check the data before pivot' , data.head(10))
					data = data.pivot_table('trans_percent_dist' , key_level , [key_dt + '_str']).reset_index()
					base_data = pd.merge(base_data, data, on = key_level , how= 'left')
				print('lets see the data with columns' , base_data.head(5) ,base_data.shape , base_data.columns )
				df_list.append(base_data)
		return df_list

	def _init_burn_dist_(self, dataframe):
		df_list = []
		for df in dataframe:
			print('entered _init_burn_dist_')
			burn_data = pd.DataFrame(df['user_cat_id'].unique()) ; burn_data.columns = ['user_cat_id']
			for key in ['f_offerdiscount']:
				for i in [2,4,8,16]:	
					print('iiiiiiiii' , i)
					data = df[(df['task_created_on_week'] > max(df['task_created_on_week'] - i ) )]
					data = data.groupby(['user_cat_id'])[key].mean().reset_index()
					data.columns = ['user_cat_id' , 'burn' + str(i) ]
					print('check the dfdfdfdffd' , data.head(5))
					burn_data = pd.merge(burn_data, data, on = 'user_cat_id' , how= 'left') 
					burn_data['burn' + str(i)].fillna(0 , inplace = True)
					print('check the dfdfdfdffd' , burn_data.head(5) )
				burn_data['burn'] = burn_data['burn2'] + burn_data['burn4']/4 + burn_data['burn8']/9 + burn_data['burn16']/16 
			df_list.append(burn_data)
		return df_list[0] , df_list[1]

	def _init_base_feature_(self,dataframe):
		df_list = []
		for df in dataframe:
			for key_level in self.dt_index:
				base_data = pd.DataFrame(df[key_level].unique()) ; base_data.columns = [key_level]
				for key in df.keys():
					if key in list(set(self.number_binary) - set(self.exception_number_features)):
						data = df.groupby([key_level]).agg({key : ['mean', 'sum']}).reset_index()
						data.columns = data.columns.droplevel(0)
						data.columns =  [key_level , key_level + key + '_mean' , key_level + key + '_sum']
						base_data = pd.merge(base_data, data, on = key_level , how= 'left')
					if key in list(set(self.number_multi) - set(self.exception_number_features)):
						data = df.groupby([key_level]).agg({key : ['mean', 'max' , 'min']}).reset_index()
						data.columns = data.columns.droplevel(0)
						data.columns =  [key_level , key_level + key + '_mean' , key_level + key + '_max' , key_level + key + '_min' ]
						base_data = pd.merge(base_data, data, on = key_level , how= 'left')
					if key in list(set(self.number_continuous) - set(self.exception_number_features)):
						data = df.groupby([key_level]).agg({key : ['mean', 'sum' , 'max' , 'min']}).reset_index()
						data.columns = data.columns.droplevel(0)
						data.columns =  [key_level , key_level + key + '_mean' , key_level + key + '_sum' , key_level + key + '_max' , key_level + key + '_min' ]
						base_data = pd.merge(base_data, data, on = key_level , how= 'left')
				print('have a look at data' ,base_data.head(5) , base_data.shape)
				df_list.append(base_data)
		return df_list

	def _init_train_val_consolidation_(self , base_data_list , target_data_list, burn_data_list):
		df_list = []
		for _list_ in self.feature_data_list:
			for i , df_level in enumerate(_list_):
				print(' i i i i' , i )
				if i  == 0 :
					base_data_list[0] = pd.merge(base_data_list[0]  ,df_level , on = 'user_id' , how = 'left')
				elif i == 1 :
					print(base_data_list[0].columns ,  df_level.columns)
					base_data_list[0]  = pd.merge(base_data_list[0]  ,df_level , on = 'user_cat_id' , how = 'left')
				elif i == 2 :
					base_data_list[1]  = pd.merge(base_data_list[1]  ,df_level , on = 'user_id' , how = 'left')
				else :
					base_data_list[1]  = pd.merge(base_data_list[1]  ,df_level , on = 'user_cat_id' , how = 'left')
		
		base_data_list[0] = pd.merge(base_data_list[0]  ,target_data_list[0] , on = 'user_cat_id' , how = 'left')
		base_data_list[1] = pd.merge(base_data_list[1]  ,target_data_list[1] , on = 'user_cat_id' , how = 'left')
		
		base_data_list[0] = pd.merge(base_data_list[0]  ,burn_data_list[0] , on = 'user_cat_id' , how = 'left')
		base_data_list[1] = pd.merge(base_data_list[1]  ,burn_data_list[1] , on = 'user_cat_id' , how = 'left')

		return base_data_list[0] , base_data_list[1]

	def _init_train_val_treat(self,dataframe):
		df_list = []
		for df in dataframe:
			df.fillna(0 , inplace = True)
			df_list.append(df)
		return df_list


	def _init_remove_features(self,dataframe):
		for df in dataframe:
			for key in self.xgb_features_exceptions:
				df[key] = 0

	def _init_model_data_(self,train , validation):
		#sim_train = train[self.sim_features] ; sim_validation = validation[self.sim_features] ; print('sim check' , sim_train.shape , sim_validation.shape )
		#sim_train.to_csv(os.path.join(args.modelsave_dir , 'sim_train_v1.csv') , index = False) 
		#sim_validation.to_csv(os.path.join(args.modelsave_dir , 'sim_validation_v1.csv') , index = False) 
		#del sim_train ; del sim_validation
		self.target_train = train[self.target] ; self.target_val = validation[self.target]
		train = train[self.features] ; validation = validation[self.features] ; print('after sim check' , train.shape , validation.shape )
		train = train.as_matrix() ; validation = validation.as_matrix()
		self.xg_train = xgb.DMatrix(train, label=self.target_train,feature_names=self.features)
		self.xg_val = xgb.DMatrix(validation, label=self.target_val,feature_names=self.features)
		return self.xg_train , self.xg_val
		
    
    
    
    
    #@author - Ankur

import pandas as pd
import numpy as np
import os
from reader import data_read
from loader import data_load
from model import xgb_model
from arguments import get_args
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print('Yo! Lets get started with Offer V2')
    args = get_args()
    print(args)

    #Call the class data_read()
    data = data_read(args.data_dir, args.data_frame ,args.date_var , args.target_var , args.idx_var , args.multi_discont , args.text_var , args.remove_var )

    #call the class data_load
    dataframe = data.data
    number_binary = data.number_binary ; number_multi = data.number_multi ; number_continuous = data.number_continuous 

    load = data_load(dataframe , number_binary , number_multi , number_continuous)

    #call the model class
    model = xgb_model(load.xg_train ,load.xg_val,load.target_train ,load.target_val , load.index_feature_train , load.index_feature_validation)
    





#@author - Ankur

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from arguments import get_args
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings("ignore")

args = get_args()

class xgb_model():
    def __init__(self,xg_train ,xg_val,target_train ,target_val,index_feature_train ,index_feature_validation ,nrounds = 250):
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'
        param["eval_metric"] = 'logloss'
        param['eta'] = 0.025
        param['max_depth'] = 6
        param['min_child_weight']=1
        param['subsample']= 0.8
        param['colsample_bytree']=0.8
        param['verbose'] = 1
        self.param = param
        self.nrounds = nrounds
        self.delta = 0.00005
        print('params' , self.param)
        self._init_model_training_iter_(xg_train,xg_val,target_train,target_val , index_feature_train ,index_feature_validation)

    def _init_model_training_iter_(self,xg_train,xg_val,target_train,target_val , index_feature_train ,index_feature_validation):
        print('start the training')
        self.model = xgb.train(self.param, xg_train, self.nrounds)
        print('training complete' , self.model)
        pred_train = self.model.predict(xg_train) ; pred_val = self.model.predict(xg_val) 
        pred_train_probabilities = pd.DataFrame(pred_train) ; pred_train_probabilities.columns = ['pred_train_probabilities']
        pred_val_probabilities = pd.DataFrame(pred_val) ; pred_val_probabilities.columns = ['pred_val_probabilities']
        self.ratio = sum(target_train)/len(target_train) + self.delta ;print('ratio is ?' , self.ratio) #; self.ratio = 0.16 ; 
        self.prob_cutoff = self._init_fin_threshohld_(pred_val) ; print('lets check the threshold' , self.prob_cutoff )
        self.pred_train =  np.where(pred_train >self.prob_cutoff , 1,0)
        self.pred_val = np.where(pred_val> self.prob_cutoff , 1, 0 )
        self.target_train = target_train.values ; self.target_val = target_val.values
        
        submit_train = pd.DataFrame(self.pred_train) ; submit_train.columns = ['pred_train']
        submit_train = pd.concat([index_feature_train ,pred_train_probabilities , submit_train] , axis = 1)
        submit_train.to_csv(os.path.join(args.save_dir , 'train_nov21.csv') , index = False) 

        submit_validation = pd.DataFrame(self.pred_val) ; submit_validation.columns = ['pred_val']
        submit_validation = pd.concat([index_feature_validation ,pred_val_probabilities , submit_validation] , axis = 1)
        submit_validation.to_csv(os.path.join(args.save_dir , 'validation_nov21.csv') , index = False) 
        print('target length' , (self.target_train) , (self.target_val) , len(self.target_train) , len(self.target_val))
        self.score_train = f1_score(self.target_train, self.pred_train)
        self.score_val = f1_score(self.target_val, self.pred_val)
        print('model results' , self.score_train , self.score_val)
        self._init_feature_imp_(self.model)
        pickle.dump(self.model , open(os.path.join(args.modelsave_dir , 'pima_nov21.pickle.dat') , 'wb' ))


    def _init_fin_threshohld_(self, preds):
        pred_sorted = sorted(preds)
        break_point = int((1-self.ratio)*len(pred_sorted))
        print(pred_sorted[break_point], break_point)
        return pred_sorted[break_point]

    def _init_feature_imp_(self,model):
        feature_imp = self.model.get_score(importance_type='gain')
        feature_imp = list(feature_imp.items()) ; feature_imp = pd.DataFrame(feature_imp , columns = ['Feature' , 'Gains'])
        feature_imp = feature_imp.sort_values(by = ['Gains'] , ascending = False)
        feature_imp.to_csv(os.path.join(args.save_dir , 'feature_imp_nov21.csv'), index=False)
        print('feature_imp' , feature_imp)

    def _init_f1_score_single_(y_true, y_pred):
        y_true = set(y_true)
        y_pred = set(y_pred)
        cross_size = len(y_true & y_pred)
        if cross_size == 0: return 0.
        p = 1. * cross_size / len(y_pred)
        r = 1. * cross_size / len(y_true)
        return 2 * p * r / (p + r)

    def f1_score(y_true, y_pred):
        return np.mean([_init_f1_score_single_(x, y) for x, y in zip(y_true, y_pred)])
        
        
        
        
        
        
        #@author - Ankur

import pandas as pd
import numpy as np
import os
import datetime
from dask import dataframe as dd 
from arguments import get_args
import warnings
warnings.filterwarnings("ignore")

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
            #data = dd.read_csv(os.path.join(self.dir ,df) , na_values = ' ', low_memory=False , assume_missing = True) ; data = data.compute()
            data = pd.read_csv(os.path.join(self.dir ,df) ,na_values = ' ', low_memory=False )
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
                    df[key + '_week'] = df[key].dt.week
                    df[key + '_day'] = df[key].dt.day
                    df[key + '_weekday'] = df[key].dt.weekday
                    df = df.sort_values([key],ascending=True)
                    print('check the df' , df.head(2))
                    number += [key + '_year' , key + '_month', key + '_week', key + '_day', key + '_weekday' ]
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
