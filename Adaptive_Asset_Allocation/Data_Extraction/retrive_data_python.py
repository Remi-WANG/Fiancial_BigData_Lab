#导入程序包及程序包重命名
pip install pymysql
import pymysql
import pandas as pd
#连接SQL数据库
conn = pymysql.connect(
                host = '172.19.3.249',    
                port = 3306,
                user ='ktruc002' ,               
passwd = '35442fed',
                db = 'cn_bond',
                charset = 'gbk'
            ) 
#提取数据库中特定的表格并输出
data = pd.read_sql('select * from  bond_info',con = conn)
print(data.values)
