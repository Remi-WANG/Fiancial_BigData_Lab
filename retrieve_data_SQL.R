#安装所需程序包
install.packages('DBI')
install.packages('RMySQL')

library(DBI)
library(RMySQL)

Sys.setlocale("LC_CTYPE", locale="Chinese")

#将R与SQL进行连接（参数已调好无需改动）
Mydb=dbConnect(MySQL(),user='ktruc002',password='35442fed', 
               dbname='cn_stock_index', host='172.19.3.249')


dbListTables(Mydb) ## displays table names

dbListFields(Mydb, 'daily_quote')  ##returns field names in the table


#赋值SQL中提取数据的语句，其中select为挑选所需表格的列名，即筛选列；
#from为选取表格语句;where为条件语句，即筛选符合条件的行；order by为排序原则

SQL_statement<- "SELECT trade_date,last
FROM daily_quote
WHERE index_code='000001'
ORDER BY trade_date DESC"

#执行语句，提取数据
dbGetQuery(Mydb,SQL_statement)

#将所提取数据赋值于变量，以便调配使用
res1 <- dbGetQuery(Mydb,SQL_statement)

dbDisconnect(Mydb)

### another example ------------

mydb=dbConnect(MySQL(),user='ktruc002',password='35442fed', 
               dbname='cn_stock_quote', host='172.19.3.249')

dbListTables(mydb) ## displays table names

dbListFields(mydb, 'daily_adjusted_quote')  ##returns field names in the table


query<-"SELECT code,day,open,high,close,volume FROM daily_adjusted_quote 
           WHERE 
           code='600000' and
           day>'2018-01-01'"


data=dbGetQuery(mydb,query)


## another way to retrieve data

rs = dbSendQuery(mydb, query)

data=fetch(rs, n=-1)


dbDisconnect(mydb)

