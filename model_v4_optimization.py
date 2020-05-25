#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import joblib
import sys
import pymssql
import pymongo
import time
import requests
import datetime as dt
from flask import g
import flask
import re


# In[12]:


#Getdata部分
def func(x):
    if 'Value' in x.keys():
        return([x['TagId'],x['Time'],x['Value']])
    else:
        return([x['TagId'],x['Time'],None])

def getdata(tags,start_time,end_time):
    formatStr = "%Y-%m-%d %H:%M:%S"
    tmObject = time.strptime(start_time+" 00:00:00", formatStr)
    tmStamp_start = str(int(time.mktime(tmObject)))
    tmObject = time.strptime(end_time+" 00:00:00", formatStr)
    tmStamp_end = str(int(time.mktime(tmObject)))
    t=int(eval("("+tags+")"))
    url='http://s-hce.hz.ds.se.com/data/tag/'+str(t)+'/0/'+tmStamp_start+'~'+tmStamp_end
    r=requests.get(url)
    text=r.text.replace("null","''")
    temp=eval(text)
    temp=pd.DataFrame(list(map(lambda x:func(x),temp)),columns=['tag_id','Time','power'])
    temp['date']=pd.to_datetime(temp.Time.apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    if 'df' not in dir():
        df=temp
    else:
        df=pd.concat([df,temp],sort=True).reset_index()
        del df['index']
    return(df)
#算法部分
def transform(df):
    length=df.columns.size
    columns=list(df.columns[0:length])
    for k in range(length):
        df=  df[df[columns[k]].notnull()]  
    return (df)  
def Polynomial_Feature(df):
    len=df.columns.size
    columns=list(df.columns[0:len])
    #df=data.drop('Date',axis=1)
    poly = PolynomialFeatures(5)
    X=poly.fit_transform(df.drop(columns[0],axis=1).values) 
    y = df[columns[0]].values
    return(X,y)
def select_Feature(X,y):
    X=SelectKBest(score_func=f_regression,k=5).fit_transform(X,y)
    return(X,y)
def healthindex(x,a,b):
    tmp1=x>=a
    tmp2=x>=b
    tmp=tmp1.astype(int)+tmp2.astype(int)
    return(tmp)


# In[13]:


Building_list=[811474,817343,817346,817348,802456,809067,801049,832076,804102]
CreateTime_list=['2019-01-01','2019-01-01','2019-01-01','2019-01-01','2018-06-01','2018-06-01','2018-04-01','2019-01-01','2019-01-01']
TriggerTime_list=['2020-03-01','2020-03-01','2020-03-01','2020-03-01','2020-03-01','2020-03-01','2019-03-01','2020-03-01','2020-03-01']
#Building_list=[801049]
#CreateTime_list=['2018-04-01']
#TriggerTime_list=['2020-03-01']

# In[10]:


#从SQLServer获取Tag相关信息
start = dt.datetime.now()
for b in range(len(Building_list)):
    BuildingId = Building_list[b]
    CreateTime= CreateTime_list[b]
    TriggerTime= TriggerTime_list[b]
    print('get argument')
    cnxn =  pymssql.connect(server='sdb2.hz.ds.se.com', user='cnrem', password='P@ssw2rd', database='energymost')
    cursor = cnxn.cursor()
    BuildingId = str(BuildingId)
    print(BuildingId)
    str_list = [BuildingId, ')']
    a = ''
    id_=a.join(str_list)
    query = ''.join(["select  pD.CustomerId,pD.DeviceId,pD.Name,pD.tagid from PopDiagnose pD join Tag T on pD.TagId=T.id", " where T.type=2 and pD.CustomerId in (select CustomerId from Hierarchy where id= ", id_])
    tag_device = pd.read_sql(query,cnxn)
    print('read sql')
    cursor.close()
    cnxn .close()
    Devicelist=list(tag_device.loc[:,'DeviceId'])
    CreateTime=str(CreateTime)
    TriggerTime=str(TriggerTime)
    if tag_device.empty:
        print('no sql data')
    else:
        #从HardCore取数
        for DeviceId in Devicelist:
            print(DeviceId)
            data=pd.DataFrame()
            DeviceTag=tag_device[tag_device['DeviceId']==DeviceId ].reset_index()
            taglist=list(DeviceTag.loc[:,'tagid'])
            for t in taglist:
                temp=getdata(str(int(t)),CreateTime,TriggerTime)
                if 'data' not in dir():
                    data=temp
                else:
                    data=pd.concat([data,temp],sort=True).reset_index()
                    del data['index']
            print('data has been extracted')
            print(data.shape)
            if data.empty:
                print('no hardcoredata ')
            else:
                DeviceTag.tagid=DeviceTag['tagid'].astype('int')
                data.tag_id=data['tag_id'].astype('int')
                databydevice=pd.merge(data,DeviceTag,left_on='tag_id',right_on='tagid').reset_index()
                del databydevice['index']
                del data
                r1 = "[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\\)(\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\+\_\-]"
                databydevice['name']=databydevice.Name.apply(lambda x:re.sub(r1,'',x))
                for k in ['-','（','）','，']:
                    databydevice['name']=databydevice.name.apply(lambda x:x.replace(k," ").strip())
                del databydevice['Name']
                databydevice.power=databydevice['power'].astype('float')
                databydevice=transform(databydevice)
                result = pd.DataFrame()
                print('data been cleaned')
                pivoted=databydevice.pivot_table( values='power', index='date', columns='name',fill_value=0, aggfunc='mean')
                pivoted.index.name = None
                pivoted['date']=pivoted.index
                if '三相电流不平衡'and '负载率' in pivoted.columns:
                    pivoted.loc[pivoted.负载率<15,'三相电流不平衡']= np.nan
                if '系统电流谐波'and '负载率' in pivoted.columns:
                    pivoted.loc[pivoted.负载率<15,'系统电流谐波']= np.nan
                if '电流谐波'and '负载率' in pivoted.columns:
                    pivoted.loc[pivoted.负载率<15,'电流谐波']= np.nan
                melted=pd.melt(pivoted,id_vars=['date'])
                result = pd.merge(melted, databydevice.loc[:,['tag_id','name']].drop_duplicates(),how='left', on=['name', 'name'])
                data2=result.copy()
                del melted
                del result
                data2.date=data2['date'].astype('str')
                data2['month']=data2.date.apply(lambda x:x.split(' ')[0].split('-')[1].split('-')[0])
                data2['year']=data2.date.apply(lambda x:x.split(' ')[0].split('-')[0])
                data2['day']=data2.date.apply(lambda x:x.split(' ')[0].split('-')[2])
                data2['hour']=data2.date.apply(lambda x:x.split(' ')[1].split(':')[0])
                data2['minutes']=data2.date.apply(lambda x:x.split(' ')[1].split(':')[1].split(':')[0])
                data2['weekday']=pd.to_datetime(data2.date).dt.weekday
                data2['ifweekday'] = data2.weekday.apply(lambda x: 1 if x<5 else 0)
                data2.month=data2['month'].astype('int')
                data2.day=data2['day'].astype('int')
                data2.hour=data2['hour'].astype('int')
                data2.minutes=data2['minutes'].astype('int')
                data2.ifweekday=data2['ifweekday'].astype('int')
                #特征工程（数据哑化）
                month_onehot = pd.get_dummies(data2.month,prefix='month',drop_first=False)
                day_onehot = pd.get_dummies(data2.day,prefix='day',drop_first=False).reset_index()
                hour_onehot = pd.get_dummies(data2.hour,prefix='hour',drop_first=False).reset_index()
                minutes_onehot = pd.get_dummies(data2.minutes,prefix='minutes',drop_first=False).reset_index()
                weekday_onehot = pd.get_dummies(data2.weekday,prefix='weekday',drop_first=False).reset_index()
                tagdata_date=pd.concat([data2.loc[:,['date','tag_id','value','ifweekday']],month_onehot],axis=1).reset_index()
                tagdata_date=pd.concat([tagdata_date,day_onehot],axis=1).reset_index()
                tagdata_date=pd.concat([tagdata_date,hour_onehot],axis=1)
                tagdata_date=pd.concat([tagdata_date,minutes_onehot],axis=1)
                tagdata_date=pd.concat([tagdata_date,weekday_onehot],axis=1)
                del data2
                del tagdata_date['index']
                del tagdata_date['level_0']
                print('dummy data')
                print(tagdata_date.shape)
                #模型训练及预测
                if tagdata_date.shape[0]>100:
                    tagdata=transform(tagdata_date)
                    tagdata.tag_id=tagdata['tag_id'].astype('int')
                    date=tagdata.loc[:,['date','tag_id']]
                    del tagdata['date']
                    list_train=[]
                    list_test=[]
                    taglist=[]
                    list_tag=tagdata.tag_id
                    list_tag=list(set(list_tag))
                    remat_train=pd.DataFrame()
                    remat_test=pd.DataFrame()
                    remat_predict=pd.DataFrame()
                    for tag_id in list_tag:
                        data_tmp=tagdata[tagdata["tag_id"]==tag_id]
                        data_tmp=data_tmp.drop(['tag_id'], axis=1)
                        date2=date[date["tag_id"]==tag_id]
                        len=data_tmp.columns.size
                        columns=list(data_tmp.columns[0:len])
                        X=data_tmp.drop(columns[0],axis=1).values
                        y = data_tmp[columns[0]].values
                        if np.size(y)>1 :
                            X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state=42)
                            model_l=LinearRegression()
                            model_l.fit(X=X_train,y=y_train)
                            list_train.insert(tag_id,r2_score(y_pred=model_l.predict(X_train),y_true=y_train))
                            list_test.insert(tag_id,r2_score(y_pred=model_l.predict(X_test),y_true=y_test))
                            taglist.insert(tag_id,tag_id)
                            tag_train=pd.DataFrame()
                            tag_test=pd.DataFrame()
                            tag_predict=pd.DataFrame()
                            tag_train['y_true']=list(y_train)
                            tag_train['pred_linear']=model_l.predict(X_train)
                            tag_train['tagid']=tag_id
                            tag_test['y_true']=list(y_test)
                            tag_test['pred_linear']=model_l.predict(X_test)
                            tag_test['tagid']=tag_id
                            tag_predict['y_true']=list(y)
                            tag_predict['pred_linear']=model_l.predict(X)
                            tag_predict['tagid']=tag_id
                            tag_predict['date']=date2.date.values
                            remat_train=pd.concat([remat_train,tag_train],axis=0)
                            remat_test=pd.concat([remat_test,tag_test],axis=0)
                            remat_predict=pd.concat([remat_predict,tag_predict],axis=0)
                            print(remat_predict.shape)
                        R2={"tagid" : taglist,"R2" : list_train}#将列表a，b转换成字典
                        tag_R2=pd.DataFrame(R2)
                        tag_regular=tag_R2.tagid[tag_R2.R2>0.5]
                        if remat_predict.empty or tag_regular.empty:
                            print(tag_regular)
                       #predict=remat_predict[remat_predict['tagid'].isin(tag_regular)]
                        else:
                        #天数据处理
                            predict=remat_predict[remat_predict['tagid'].isin(tag_regular)]
                            predict1=predict.copy()
                            predict1['Time']=predict1.date.apply(lambda x:x.split(' ')[1])
                            predict1=predict1.groupby(['tagid','Time']).mean().reset_index()
                        #周数据处理
                            predict2=predict.copy()
                            predict2['Time']=predict.date.apply(lambda x:x.split(' ')[1])
                            predict2['WeekDay']=pd.to_datetime(predict2.date).dt.weekday
                            del predict2['date']
                            predict2=predict2.groupby(['tagid','WeekDay','Time']).mean().reset_index()
                         #给出各tag临界值和风险指数
                            predict1_heathindex=pd.DataFrame()
                            predict2_heathindex=pd.DataFrame()
                            result_day=pd.DataFrame()
                            result_week=pd.DataFrame()
                            predict1['residual']=abs(predict1.y_true-predict1.pred_linear)
                            predict2['residual']=abs(predict2.y_true-predict2.pred_linear)
                            del predict
                            for tagid in tag_regular:
                                predict1_heathindex=predict1[predict1["tagid"]==tagid]
                                predict2_heathindex=predict2[predict2["tagid"]==tagid]
                                predict1_heathindex['criterion1']=np.percentile(predict1_heathindex['residual'], 75)
                                predict1_heathindex['criterion2']=np.percentile(predict1_heathindex['residual'], 90)
                                predict2_heathindex['criterion1']=np.percentile(predict2_heathindex['residual'], 75)
                                predict2_heathindex['criterion2']=np.percentile(predict2_heathindex['residual'], 90)
                                result_day=pd.concat([result_day,predict1_heathindex],axis=0)
                                result_week=pd.concat([result_week,predict2_heathindex],axis=0)
                            result_day['healthindex']=healthindex(result_day['residual'].values,result_day['criterion1'][0],result_day['criterion2'][0])
                            result_week['healthindex']=healthindex(result_week['residual'].values,result_week['criterion1'][0],result_week['criterion2'][0])
                            result_day=pd.merge(result_day,tag_R2,left_on='tagid',right_on='tagid').reset_index()
                            result_week=pd.merge(result_week,tag_R2,left_on='tagid',right_on='tagid').reset_index()
                            del result_day['index']
                            del result_week['index']
                            result_day=result_day.rename(columns={'tagid': 'TagId','y_true':'Value','pred_linear':'FittedValue','residual':'ResidualValue','criterion1':'Threhold1','criterion2':'Threhold2','healthindex':'RiskIndex','R2':'FittedGoodness'})
                            result_week=result_week.rename(columns={'tagid': 'TagId','y_true':'Value','pred_linear':'FittedValue','residual':'ResidualValue','criterion1':'Threhold1','criterion2':'Threhold2','healthindex':'RiskIndex','R2':'FittedGoodness'})
                            result_day['TriggerTime']=TriggerTime
                            result_week['TriggerTime']=TriggerTime
                            result_day['BuildingId']=BuildingId
                            result_week['BuildingId']=BuildingId
                            del predict1
                            del predict2
                            print('health index finished')
                            mongo_connectstr='mongodb://rem:Bnz2U568doW79F@dds-bp1da4fc20ed51641.mongodb.rds.aliyuncs.com:3717,dds-bp1da4fc20ed51642.mongodb.rds.aliyuncs.com:3717/energymost?replicaSet=mgset-12719007'
                            client = pymongo.MongoClient(mongo_connectstr)
                            # database
                            db_mongo = client.get_database()
                            print(TriggerTime)
                            print(result_day.head())
                            print(result_week.head())
                            print('get mongodb database')
                            # collection
                            Pop_FittedCurveDailyData=db_mongo["Pop_FittedCurveDailyData"]
                            Pop_FittedCurveWeeklyData=db_mongo["Pop_FittedCurveWeeklyData"]
                            result_day_dict = result_day.to_dict("records")
                            result_week_dict = result_week.to_dict("records")
                            db_mongo.Pop_FittedCurveDailyData.delete_many({"BuildingId": BuildingId,"TriggerTime": TriggerTime})
                            db_mongo.Pop_FittedCurveWeeklyData.delete_many({"BuildingId": BuildingId,"TriggerTime": TriggerTime})
                            db_mongo.Pop_FittedCurveDailyData.insert_many(result_day_dict)
                            db_mongo.Pop_FittedCurveWeeklyData.insert_many(result_week_dict)
                            del result_day
                            del result_week
                            del result_day_dict
                            del result_week_dict
                            print('data been saved')
end = dt.datetime.now()
print(end-start)

