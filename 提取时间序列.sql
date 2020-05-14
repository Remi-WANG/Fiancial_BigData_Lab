SHOW OPEN TABLES FROM cn_stock_quote WHERE `in_use`!=0;
USE `cn_stock_quote`;
SELECT code,shortname,trade_date,EXCHANGE,pre_close,OPEN,volume,high,low,LAST
FROM daily_quote
WHERE CODE='000001';