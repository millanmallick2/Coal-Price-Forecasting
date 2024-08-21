create database project;
use project;
select * from coaldata;
SHOW DATABASES;

#EDA
#1st business Moments
show columns from coaldata;
#MEAN
select avg(Coal_RB_4800_FOB_London_Close_USD), avg(Coal_RB_5500_FOB_London_Close_USD), avg(Coal_RB_5700_FOB_London_Close_USD),
avg(Coal_RB_6000_FOB_CurrentWeek_Avg_USD), avg(Coal_India_5500_CFR_London_Close_USD), avg(Price_WTI), avg(Price_Brent_Oil),
avg(Price_Dubai_Brent_Oil), avg(Price_ExxonMobil), avg(Price_Shenhua), avg(Price_All_Share), avg(Price_Mining),
avg(Price_LNG_Japan_Korea_Marker_PLATTS), avg(Price_ZAR_USD), avg(Price_Natural_Gas), avg(Price_ICE), avg(Price_Dutch_TTF),
avg(Price_Indian_en_exg_rate)  from coaldata;

#Median
#1. Coal_RB_4800_FOB_London_Close_USD
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Coal_RB_4800_FOB_London_Close_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Coal_RB_4800_FOB_London_Close_USD FROM coaldata ORDER BY Coal_RB_4800_FOB_London_Close_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));

#2. Coal_RB_5500_FOB_London_Close_USD
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Coal_RB_5500_FOB_London_Close_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Coal_RB_5500_FOB_London_Close_USD FROM coaldata ORDER BY Coal_RB_5500_FOB_London_Close_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));
 
 #3.Coal_RB_5700_FOB_London_Close_USD
 SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Coal_RB_5700_FOB_London_Close_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Coal_RB_5700_FOB_London_Close_USD FROM coaldata ORDER BY Coal_RB_5700_FOB_London_Close_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
 
#4.Coal_RB_6000_FOB_CurrentWeek_Avg_USD
  SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Coal_RB_6000_FOB_CurrentWeek_Avg_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Coal_RB_6000_FOB_CurrentWeek_Avg_USD FROM coaldata ORDER BY Coal_RB_6000_FOB_CurrentWeek_Avg_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#5.Coal_India_5500_CFR_London_Close_USD
 SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Coal_India_5500_CFR_London_Close_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Coal_India_5500_CFR_London_Close_USD FROM coaldata ORDER BY Coal_India_5500_CFR_London_Close_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
 
 #6.Price_WTI
 SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_WTI AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_WTI FROM coaldata ORDER BY Price_WTI) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#7.Price_Brent_Oil
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Brent_Oil AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Brent_Oil FROM coaldata ORDER BY Price_Brent_Oil) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#8. Price_Dubai_Brent_Oil
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Dubai_Brent_Oil AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Dubai_Brent_Oil FROM coaldata ORDER BY Price_Dubai_Brent_Oil) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#9.Price_ExxonMobil
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_ExxonMobil AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_ExxonMobil FROM coaldata ORDER BY Price_ExxonMobil) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#10.Price_Shenhua
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Shenhua AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Shenhua FROM coaldata ORDER BY Price_Shenhua) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#11.  Price_All_Share  
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_All_Share AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_All_Share FROM coaldata ORDER BY Price_All_Share) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#12. Price_Mining
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Mining AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Mining FROM coaldata ORDER BY Price_Mining) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#13.Price_LNG_Japan_Korea_Marker_PLATTS
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_LNG_Japan_Korea_Marker_PLATTS AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_LNG_Japan_Korea_Marker_PLATTS FROM coaldata ORDER BY Price_LNG_Japan_Korea_Marker_PLATTS) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#14.Price_ZAR_USD
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_ZAR_USD AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_ZAR_USD FROM coaldata ORDER BY Price_ZAR_USD) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#15.Price_Natural_Gas
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Natural_Gas AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Natural_Gas FROM coaldata ORDER BY Price_Natural_Gas) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   

#16.Price_ICE
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_ICE AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_ICE FROM coaldata ORDER BY Price_ICE) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#17.Price_Dutch_TTF
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Dutch_TTF AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Dutch_TTF FROM coaldata ORDER BY Price_Dutch_TTF) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
    
#18.Price_Indian_en_exg_rate
SELECT 
    AVG(Val) AS Median
FROM (
    SELECT 
        Price_Indian_en_exg_rate AS Val,
        @rownum := @rownum + 1 AS row_num,
        @total_rows := @rownum
    FROM 
        (SELECT Price_Indian_en_exg_rate FROM coaldata ORDER BY Price_Indian_en_exg_rate) AS ordered_data,
        (SELECT @rownum := 0) r
) AS sorted_data
WHERE
    row_num IN (FLOOR((@total_rows + 1) / 2), CEIL((@total_rows + 1) / 2));   
 
#mode
select Coal_RB_4800_FOB_London_Close_USD from coaldata group by Coal_RB_4800_FOB_London_Close_USD  order by count(*) desc limit 2;
select Coal_RB_5500_FOB_London_Close_USD from coaldata group by Coal_RB_5500_FOB_London_Close_USD  order by count(*) desc limit 2;
select Coal_RB_5700_FOB_London_Close_USD from coaldata group by Coal_RB_5700_FOB_London_Close_USD  order by count(*) desc limit 2;
select Coal_RB_6000_FOB_CurrentWeek_Avg_USD from coaldata group by Coal_RB_6000_FOB_CurrentWeek_Avg_USD  order by count(*) desc limit 2;
select Coal_India_5500_CFR_London_Close_USD from coaldata group by Coal_India_5500_CFR_London_Close_USD  order by count(*) desc limit 2;
select Price_WTI from coaldata group by Price_WTI  order by count(*) desc limit 2;
select Price_Brent_Oil from coaldata group by Price_Brent_Oil  order by count(*) desc limit 2;
select Price_Dubai_Brent_Oil from coaldata group by Price_Dubai_Brent_Oil  order by count(*) desc limit 2;
select Price_ExxonMobil from coaldata group by Price_ExxonMobil order by count(*) desc limit 2;
select Price_Shenhua from coaldata group by Price_Shenhua  order by count(*) desc limit 2;
select Price_All_Share from coaldata group by Price_All_Share  order by count(*) desc limit 2;
select Price_Mining from coaldata group by Price_Mining  order by count(*) desc limit 2;
select Price_LNG_Japan_Korea_Marker_PLATTS from coaldata group by Price_LNG_Japan_Korea_Marker_PLATTS order by count(*) desc limit 2;
select Price_ZAR_USD from coaldata group by Price_ZAR_USD  order by count(*) desc limit 2;
select Price_Natural_Gas from coaldata group by Price_Natural_Gas order by count(*) desc limit 2;
select Price_ICE from coaldata group by Price_ICE order by count(*) desc limit 2;
select Price_Dutch_TTF from coaldata group by Price_Dutch_TTF  order by count(*) desc limit 2;
select Price_Indian_en_exg_rate from coaldata group by Price_Indian_en_exg_rate order by count(*) desc limit 2;

#2nd BUsiness Moments
#VAriance
select variance(Coal_RB_4800_FOB_London_Close_USD), variance(Coal_RB_5500_FOB_London_Close_USD), variance(Coal_RB_5700_FOB_London_Close_USD),
variance(Coal_RB_6000_FOB_CurrentWeek_Avg_USD), variance(Coal_India_5500_CFR_London_Close_USD), variance(Price_WTI), variance(Price_Brent_Oil),
variance(Price_Dubai_Brent_Oil), variance(Price_ExxonMobil), variance(Price_Shenhua), variance(Price_All_Share), variance(Price_Mining),
variance(Price_LNG_Japan_Korea_Marker_PLATTS), variance(Price_ZAR_USD), variance(Price_Natural_Gas), variance(Price_ICE), variance(Price_Dutch_TTF),
variance(Price_Indian_en_exg_rate)  from coaldata;

#STANDARD DEVIATION
select stddev(Coal_RB_4800_FOB_London_Close_USD), stddev(Coal_RB_5500_FOB_London_Close_USD), stddev(Coal_RB_5700_FOB_London_Close_USD),
stddev(Coal_RB_6000_FOB_CurrentWeek_Avg_USD), stddev(Coal_India_5500_CFR_London_Close_USD), stddev(Price_WTI), stddev(Price_Brent_Oil),
stddev(Price_Dubai_Brent_Oil), stddev(Price_ExxonMobil), stddev(Price_Shenhua), stddev(Price_All_Share), stddev(Price_Mining),
stddev(Price_LNG_Japan_Korea_Marker_PLATTS), stddev(Price_ZAR_USD), stddev(Price_Natural_Gas), stddev(Price_ICE), stddev(Price_Dutch_TTF),
stddev(Price_Indian_en_exg_rate)  from coaldata;

#3RD BUSINESS MOMENT
#skewness
#1. Coal_RB_4800_FOB_London_Close_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Coal_RB_4800_FOB_London_Close_USD) AS stddev,
        AVG(Coal_RB_4800_FOB_London_Close_USD) AS mean,
        SUM(POWER(Coal_RB_4800_FOB_London_Close_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Coal_RB_4800_FOB_London_Close_USD,
            AVG(Coal_RB_4800_FOB_London_Close_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#2. Coal_RB_5500_FOB_London_Close_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Coal_RB_5500_FOB_London_Close_USD) AS stddev,
        AVG(Coal_RB_5500_FOB_London_Close_USD) AS mean,
        SUM(POWER(Coal_RB_5500_FOB_London_Close_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Coal_RB_5500_FOB_London_Close_USD,
            AVG(Coal_RB_5500_FOB_London_Close_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
 
 #3.Coal_RB_5700_FOB_London_Close_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Coal_RB_5700_FOB_London_Close_USD) AS stddev,
        AVG(Coal_RB_5700_FOB_London_Close_USD) AS mean,
        SUM(POWER(Coal_RB_5700_FOB_London_Close_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Coal_RB_5700_FOB_London_Close_USD,
            AVG(Coal_RB_5700_FOB_London_Close_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
 
#4.Coal_RB_6000_FOB_CurrentWeek_Avg_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) AS stddev,
        AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) AS mean,
        SUM(POWER(Coal_RB_6000_FOB_CurrentWeek_Avg_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Coal_RB_6000_FOB_CurrentWeek_Avg_USD,
            AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
    
#5.Coal_India_5500_CFR_London_Close_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Coal_India_5500_CFR_London_Close_USD) AS stddev,
        AVG(Coal_India_5500_CFR_London_Close_USD) AS mean,
        SUM(POWER(Coal_India_5500_CFR_London_Close_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Coal_India_5500_CFR_London_Close_USD,
            AVG(Coal_India_5500_CFR_London_Close_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
 
 #6.Price_WTI
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_WTI) AS stddev,
        AVG(Price_WTI) AS mean,
        SUM(POWER(Price_WTI - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_WTI,
            AVG(Price_WTI) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived; 

#7.Price_Brent_Oil
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Brent_Oil) AS stddev,
        AVG(Price_Brent_Oil) AS mean,
        SUM(POWER(Price_Brent_Oil - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Brent_Oil,
            AVG(Price_Brent_Oil) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
    
#8. Price_Dubai_Brent_Oil
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Dubai_Brent_Oil) AS stddev,
        AVG(Price_Dubai_Brent_Oil) AS mean,
        SUM(POWER(Price_Dubai_Brent_Oil - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Dubai_Brent_Oil,
            AVG(Price_Dubai_Brent_Oil) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
    
#9.Price_ExxonMobil
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_ExxonMobil) AS stddev,
        AVG(Price_ExxonMobil) AS mean,
        SUM(POWER(Price_ExxonMobil - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_ExxonMobil,
            AVG(Price_ExxonMobil) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#10.Price_Shenhua
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Shenhua) AS stddev,
        AVG(Price_Shenhua) AS mean,
        SUM(POWER(Price_Shenhua - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Shenhua,
            AVG(Price_Shenhua) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#11.  Price_All_Share  
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_All_Share) AS stddev,
        AVG(Price_All_Share) AS mean,
        SUM(POWER(Price_All_Share - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_All_Share,
            AVG(Price_All_Share) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
    
#12. Price_Mining
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Mining) AS stddev,
        AVG(Price_Mining) AS mean,
        SUM(POWER(Price_Mining - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Mining,
            AVG(Price_Mining) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#13.Price_LNG_Japan_Korea_Marker_PLATTS
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_LNG_Japan_Korea_Marker_PLATTS) AS stddev,
        AVG(Price_LNG_Japan_Korea_Marker_PLATTS) AS mean,
        SUM(POWER(Price_LNG_Japan_Korea_Marker_PLATTS - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_LNG_Japan_Korea_Marker_PLATTS,
            AVG(Price_LNG_Japan_Korea_Marker_PLATTS) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived; 

#14.Price_ZAR_USD
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_ZAR_USD) AS stddev,
        AVG(Price_ZAR_USD) AS mean,
        SUM(POWER(Price_ZAR_USD - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_ZAR_USD,
            AVG(Price_ZAR_USD) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#15.Price_Natural_Gas
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Natural_Gas) AS stddev,
        AVG(Price_Natural_Gas) AS mean,
        SUM(POWER(Price_Natural_Gas - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Natural_Gas,
            AVG(Price_Natural_Gas) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;

#16.Price_ICE
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_ICE) AS stddev,
        AVG(Price_ICE) AS mean,
        SUM(POWER(Price_ICE - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_ICE,
            AVG(Price_ICE) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived; 
    
#17.Price_Dutch_TTF
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Dutch_TTF) AS stddev,
        AVG(Price_Dutch_TTF) AS mean,
        SUM(POWER(Price_Dutch_TTF - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Dutch_TTF,
            AVG(Price_Dutch_TTF) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
    
#18.Price_Indian_en_exg_rate
SELECT 
    n,
    sum_cubed_deviations / (n * POWER(stddev, 3)) AS skewness
FROM (
    SELECT
        COUNT(*) AS n,
        STDDEV_POP(Price_Indian_en_exg_rate) AS stddev,
        AVG(Price_Indian_en_exg_rate) AS mean,
        SUM(POWER(Price_Indian_en_exg_rate - mean_val, 3)) AS sum_cubed_deviations
    FROM (
        SELECT
            Price_Indian_en_exg_rate,
            AVG(Price_Indian_en_exg_rate) OVER () AS mean_val
        FROM
            coaldata
    ) AS subquery
) AS derived;
 
 
#4TH BUSINESS MOMENT
#Kurtosis
#1 Coal_RB_4800_FOB_London_Close_USD
CREATE TEMPORARY TABLE kurtosis_coal_4800 AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Coal_RB_4800_FOB_London_Close_USD) AS stddev,
    AVG(Coal_RB_4800_FOB_London_Close_USD) AS mean,
    SUM(POWER(Coal_RB_4800_FOB_London_Close_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Coal_RB_4800_FOB_London_Close_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_RB_4800_FOB_London_Close_USD,
        (SELECT AVG(Coal_RB_4800_FOB_London_Close_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_coal_4800;
    
#2. Coal_RB_5500_FOB_London_Close_USD
CREATE TEMPORARY TABLE kurtosis_coal_5500 AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Coal_RB_5500_FOB_London_Close_USD) AS stddev,
    AVG(Coal_RB_5500_FOB_London_Close_USD) AS mean,
    SUM(POWER(Coal_RB_5500_FOB_London_Close_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Coal_RB_5500_FOB_London_Close_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_RB_5500_FOB_London_Close_USD,
        (SELECT AVG(Coal_RB_5500_FOB_London_Close_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_coal_5500;
    
#3.Coal_RB_5700_FOB_London_Close_USD
CREATE TEMPORARY TABLE kurtosis_coal_5700 AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Coal_RB_5700_FOB_London_Close_USD) AS stddev,
    AVG(Coal_RB_5700_FOB_London_Close_USD) AS mean,
    SUM(POWER(Coal_RB_5700_FOB_London_Close_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Coal_RB_5700_FOB_London_Close_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_RB_5700_FOB_London_Close_USD,
        (SELECT AVG(Coal_RB_5700_FOB_London_Close_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_coal_5700;
    
#4.Coal_RB_6000_FOB_CurrentWeek_Avg_USD
CREATE TEMPORARY TABLE kurtosis_coal_6000 AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) AS stddev,
    AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) AS mean,
    SUM(POWER(Coal_RB_6000_FOB_CurrentWeek_Avg_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Coal_RB_6000_FOB_CurrentWeek_Avg_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_RB_6000_FOB_CurrentWeek_Avg_USD,
        (SELECT AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_coal_6000;
 
#5.Coal_India_5500_CFR_London_Close_USD
CREATE TEMPORARY TABLE kurtosis_Coal_India_5500_CFR_London_Close_USD AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Coal_India_5500_CFR_London_Close_USD) AS stddev,
    AVG(Coal_India_5500_CFR_London_Close_USD) AS mean,
    SUM(POWER(Coal_India_5500_CFR_London_Close_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Coal_India_5500_CFR_London_Close_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_India_5500_CFR_London_Close_USD,
        (SELECT AVG(Coal_India_5500_CFR_London_Close_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Coal_India_5500_CFR_London_Close_USD;
 
 #6.Price_WTI
CREATE TEMPORARY TABLE kurtosis_Price_WTI AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_WTI) AS stddev,
    AVG(Price_WTI) AS mean,
    SUM(POWER(Price_WTI - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_WTI - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_WTI,
        (SELECT AVG(Price_WTI) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_WTI;

#7.Price_Brent_Oil
CREATE TEMPORARY TABLE kurtosis_Price_Brent_Oil AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Brent_Oil) AS stddev,
    AVG(Price_Brent_Oil) AS mean,
    SUM(POWER(Price_Brent_Oil - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Brent_Oil - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Brent_Oil,
        (SELECT AVG(Price_Brent_Oil) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
     SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Brent_Oil;
    
#8. Price_Dubai_Brent_Oil
CREATE TEMPORARY TABLE kurtosis_Price_Dubai_Brent_Oil AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Dubai_Brent_Oil) AS stddev,
    AVG(Price_Dubai_Brent_Oil) AS mean,
    SUM(POWER(Price_Dubai_Brent_Oil - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Dubai_Brent_Oil - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Dubai_Brent_Oil,
        (SELECT AVG(Price_Dubai_Brent_Oil) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Dubai_Brent_Oil;
    
#9.Price_ExxonMobil
CREATE TEMPORARY TABLE kurtosis_Price_ExxonMobil AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_ExxonMobil) AS stddev,
    AVG(Price_ExxonMobil) AS mean,
    SUM(POWER(Price_ExxonMobil - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_ExxonMobil - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_ExxonMobil,
        (SELECT AVG(Price_ExxonMobil) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_ExxonMobil;

#10.Price_Shenhua
CREATE TEMPORARY TABLE kurtosis_Price_Shenhua AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Shenhua) AS stddev,
    AVG(Price_Shenhua) AS mean,
    SUM(POWER(Price_Shenhua - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Shenhua - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Shenhua,
        (SELECT AVG(Price_Shenhua) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Shenhua;
    
#11.  Price_All_Share  
CREATE TEMPORARY TABLE kurtosis_Price_All_Share AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_All_Share) AS stddev,
    AVG(Price_All_Share) AS mean,
    SUM(POWER(Price_All_Share - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_All_Share - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_All_Share,
        (SELECT AVG(Price_All_Share) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
   SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_All_Share;
    
#12. Price_Mining
CREATE TEMPORARY TABLE kurtosis_Price_Mining AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Mining) AS stddev,
    AVG(Price_Mining) AS mean,
    SUM(POWER(Price_Mining - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Mining - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Mining,
        (SELECT AVG(Price_Mining) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Mining;
    
#13.Price_LNG_Japan_Korea_Marker_PLATTS
CREATE TEMPORARY TABLE kurtosis_Price_LNG_Japan_Korea_Marker_PLATTS AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_LNG_Japan_Korea_Marker_PLATTS) AS stddev,
    AVG(Price_LNG_Japan_Korea_Marker_PLATTS) AS mean,
    SUM(POWER(Price_LNG_Japan_Korea_Marker_PLATTS - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_LNG_Japan_Korea_Marker_PLATTS - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_LNG_Japan_Korea_Marker_PLATTS,
        (SELECT AVG(Price_LNG_Japan_Korea_Marker_PLATTS) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_LNG_Japan_Korea_Marker_PLATTS;
    
#14.Price_ZAR_USD
CREATE TEMPORARY TABLE kurtosis_Price_ZAR_USD AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_ZAR_USD) AS stddev,
    AVG(Price_ZAR_USD) AS mean,
    SUM(POWER(Price_ZAR_USD - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_ZAR_USD - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_ZAR_USD,
        (SELECT AVG(Price_ZAR_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_ZAR_USD;
    
#15.Price_Natural_Gas
CREATE TEMPORARY TABLE kurtosis_Price_Natural_Gas AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Natural_Gas) AS stddev,
    AVG(Price_Natural_Gas) AS mean,
    SUM(POWER(Price_Natural_Gas - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Natural_Gas - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Natural_Gas,
        (SELECT AVG(Price_Natural_Gas) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Natural_Gas;
    
#16.Price_ICE
CREATE TEMPORARY TABLE kurtosis_Price_ICE AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_ICE) AS stddev,
    AVG(Price_ICE) AS mean,
    SUM(POWER(Price_ICE - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_ICE - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Coal_RB_5700_FOB_London_Close_USD,
        (SELECT AVG(Coal_RB_5700_FOB_London_Close_USD) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
  SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_ICE;
    
#17.Price_Dutch_TTF
CREATE TEMPORARY TABLE kurtosis_Price_Dutch_TTF AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Dutch_TTF) AS stddev,
    AVG(Price_Dutch_TTF) AS mean,
    SUM(POWER(Price_Dutch_TTF - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Dutch_TTF - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Dutch_TTF,
        (SELECT AVG(Price_Dutch_TTF) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Dutch_TTF;
    
#18.Price_Indian_en_exg_rate
CREATE TEMPORARY TABLE kurtosis_Price_Indian_en_exg_rate AS
SELECT
    COUNT(*) AS n,
    STDDEV_POP(Price_Indian_en_exg_rate) AS stddev,
    AVG(Price_Indian_en_exg_rate) AS mean,
    SUM(POWER(Price_Indian_en_exg_rate - mean_val, 4)) AS sum_quartic_deviations,
    SUM(POWER(Price_Indian_en_exg_rate - mean_val, 2)) AS sum_squared_deviations
FROM (
    SELECT
        Price_Indian_en_exg_rate,
        (SELECT AVG(Price_Indian_en_exg_rate) FROM coaldata) AS mean_val
    FROM
        coaldata
) AS subquery;
 SELECT
    n,
    ((n * (n + 1) * sum_quartic_deviations - 3 * POWER(sum_squared_deviations, 2)) / ((n - 1) * (n - 2) * (n - 3) * POWER(stddev, 4))) - 3 AS kurtosis
FROM
    kurtosis_Price_Indian_en_exg_rate;




























