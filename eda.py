import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

from utils import UnnamedDrop, ExtractDates
from ploter import year_column_bar_plot, yearly_heatmap, similarity_matrix_plot

### Loading the data
train = pd.read_csv('data/train.csv', index_col=0)
stores = pd.read_csv('data/stores.csv', index_col=0)

### Seperating X and y
train_X, train_y = train.loc[:, train.columns != 'sales'], train['sales']
train_X = train_X.join(stores.set_index('store_id'), on='store_id')

### Creating pipline
pipe = Pipeline([('drop', UnnamedDrop()),
                 ('dates', ExtractDates()),
                 ])
train_X = pipe.fit_transform(train_X)
train_X['sales'] = train_y

### Yearly sample count
df_sales_year_count = train_X[['sales', 'year']].groupby('year').count().reset_index()
ax = df_sales_year_count.plot.bar(x='year', y='sales', figsize=(8,7))
ax.set_title('Count of Yearly Samples', fontsize=20)
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
plt.show()
plt.close()

### Average Yearly Sales
df_sales_year = train_X[['sales', 'year']].groupby('year').mean().reset_index()
ax = df_sales_year.plot.bar(x='year', y='sales', figsize=(8,7))
ax.set_title('Average Yearly Sales', fontsize=20)
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('Mean Sales', fontsize=15)
plt.show()


### Average Weekly Sales
df_sales_woy = train_X[['sales', 'woy']].groupby('woy').mean().reset_index()
ax = df_sales_woy.plot.bar(x='woy', y='sales', figsize=(12,8))
ax.set_title('Average Weekly Sales', fontsize=20)
ax.set_xlabel('Week', fontsize=15)
ax.set_ylabel('Mean Sales', fontsize=15)
ax.tick_params(axis='x', rotation=45, labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xticks(df_sales_woy.index[::2])
plt.show()

### Average Daily Sales
df_sales_dow = train_X[['sales', 'dow']].groupby('dow').mean().reset_index()
ax = df_sales_dow.plot.bar(x='dow', y='sales')
ax.set_title('Average Daily Sales', fontsize=20)
ax.set_xlabel('Day', fontsize=15)
ax.set_ylabel('Mean Sales', fontsize=15)
plt.show()


### Average Yearly Promotion Invested
df_sales_year = train_X[['tot_promoted', 'year']].groupby('year').mean().reset_index()
ax = df_sales_year.plot.bar(x='year', y='tot_promoted', figsize=(7,7))
ax.set_title('Average Yearly Promotion Invested', fontsize=20)
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('Mean Promotion', fontsize=15)
plt.show()


### Average Weakly Promotion Invested
df_sales_woy = train_X[['tot_promoted', 'woy']].groupby('woy').mean().reset_index()
ax = df_sales_woy.plot.bar(x='woy', y='tot_promoted')
ax.set_title('Average Weakly Promotion Invested', fontsize=20)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xticks(df_sales_woy.index[::2])
plt.show()

### Average Daily Promotion Invested
df_sales_dow = train_X[['tot_promoted', 'dow']].groupby('dow').mean().reset_index()
ax = df_sales_dow.plot.bar(x='dow', y='tot_promoted')
ax.set_title('Average Daily Promotion Invested', fontsize=20)
ax.set_xlabel('Day', fontsize=15)
ax.set_ylabel('Mean Promotion', fontsize=15)
plt.show()


### Correlation Matrix
corr = train_X[['sales','tot_promoted','year', 'woy', 'dow']].corr()
mask = np.tril(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, mask=mask)
ax.set_title('Correlation Matrix', fontsize=20)
plt.show()
plt.close()



''' Yearly Sales '''
### Yearly Average Sales per dow
year_column_bar_plot(train_X, value='sales', column='dow')
### Yearly Average Sales per woy
year_column_bar_plot(train_X, value='sales', column='woy')
### Yearly Average Sales per sku_category
year_column_bar_plot(train_X, value='sales', column='sku_category')
### Yearly Average Sales per store_id
year_column_bar_plot(train_X, value='sales', column='store_id')
### Yearly Average Sales per group
year_column_bar_plot(train_X, value='sales', column='group')
### Yearly Average Sales per province
year_column_bar_plot(train_X, value='sales', column='province')
### Yearly Average Sales per geo
year_column_bar_plot(train_X, value='sales', column='geo')
### Yearly Average Sales per type
year_column_bar_plot(train_X, value='sales', column='type')

''' Yearly Promotions'''
### Yearly Average Promotion per dow
year_column_bar_plot(train_X, value='tot_promoted', column='dow')
### Yearly Average Promotion per woy
year_column_bar_plot(train_X, value='tot_promoted', column='woy')
### Yearly Average Promotion per sku_category
year_column_bar_plot(train_X, value='tot_promoted', column='sku_category')
### Yearly Average Promotion per store_id
year_column_bar_plot(train_X, value='tot_promoted', column='store_id')
### Yearly Average Promotion per group
year_column_bar_plot(train_X, value='tot_promoted', column='group')
### Yearly Average Promotion per province
year_column_bar_plot(train_X, value='tot_promoted', column='province')
### Yearly Average Promotion per geo
year_column_bar_plot(train_X, value='tot_promoted', column='geo')
### Yearly Average Promotion per type
year_column_bar_plot(train_X, value='tot_promoted', column='type')



'''Heatmaps'''
### Yearly Sales Per store_id, sku_category
similarity_matrix_sales = yearly_heatmap(train_X, values='sales', index='store_id', columns='sku_category')
# Cosine Similarity Sales by store_id, sku_category
similarity_matrix_plot(similarity_matrix_sales, 'Sales \nby store_id, sku_category')

### Yearly Promotions Per store_id, sku_category
similarity_matrix_sales = yearly_heatmap(train_X, values='tot_promoted', index='store_id', columns='sku_category')
# Cosine Similarity Promotions by store_id, sku_category
similarity_matrix_plot(similarity_matrix_sales, 'Total Promoted \nby store_id, sku_category')


### Yearly Sales Per store_id, dow
yearly_heatmap(train_X, values='sales', index='store_id', columns='dow', similarity_heatmap=False)
### Yearly Promotions Per store_id, dow
yearly_heatmap(train_X, values='tot_promoted', index='store_id', columns='dow', similarity_heatmap=False)

### Yearly Sales Per store_id, woy
yearly_heatmap(train_X, values='sales', index='store_id', columns='woy', similarity_heatmap=False )
### Yearly Promotions Per store_id, woy
yearly_heatmap(train_X, values='tot_promoted', index='store_id', columns='woy', similarity_heatmap=False)


yearly_heatmap(train_X, values='sales', index='store_id', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='sku_category', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='geo', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='type', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='dow', columns='group', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='woy', columns='group', similarity_heatmap=False)

yearly_heatmap(train_X, values='sales', index='store_id', columns='type', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='sku_category', columns='type', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='geo', columns='type', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='type', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='dow', columns='type', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='woy', columns='type', similarity_heatmap=False)

yearly_heatmap(train_X, values='sales', index='store_id', columns='dow', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='sku_category', columns='dow', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='geo', columns='dow', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='dow', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='woy', columns='dow', similarity_heatmap=False)

yearly_heatmap(train_X, values='sales', index='store_id', columns='sku_category', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='geo', columns='sku_category', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='sku_category', similarity_heatmap=False)

yearly_heatmap(train_X, values='sales', index='geo', columns='store_id', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='store_id', similarity_heatmap=False)
yearly_heatmap(train_X, values='sales', index='province', columns='geo', similarity_heatmap=False)




