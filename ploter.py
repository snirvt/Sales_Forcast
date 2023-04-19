
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def year_column_bar_plot(X_df, value='tot_promoted', column = 'dow'):
    df_promoted_year_dow = X_df[[value, column, 'year']].groupby(['year', column]).mean().reset_index()
    # Get unique years from the DataFrame
    years = X_df['year'].unique()
    # Create a figure and subplots for each year
    fig, axs = plt.subplots(nrows=len(years), figsize=(10, 10))
    vmax = df_promoted_year_dow[value].max()
    # Loop through each year and plot the data
    for i, year in enumerate(np.sort(years)):
        # Get the data for the current year
        data = df_promoted_year_dow[df_promoted_year_dow['year'] == year]
        axs[i].bar(data[column], data[value], label=f'{year}')
        axs[i].set_ylim([0, vmax*1.05])
        if i == 0:
            axs[i].set_title(f'Yearly Average {value} per {column}', fontsize=20)
        if i == 2:
            axs[i].set_ylabel(f'Mean {value}', fontsize=20)
        if i == 4:
            axs[i].set_xlabel(f'{column}', fontsize=20)
        axs[i].legend()
    plt.show()
    plt.close()






def yearly_heatmap(X_df, values='sales', index='store_id', columns='sku_category', similarity_heatmap=True):
    ### plotting store id vs category
    year_groups = X_df.groupby('year')
    # set up the subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(year_groups), figsize=(20, 7))
    pivot_list = []
    vmax, vmin = float('-inf'), float('inf')
    # loop through each year and create a separate heatmap
    for i, (year, year_data) in enumerate(year_groups):
        pivot_df = pd.pivot_table(year_data, values=values, index=index, columns=columns, aggfunc='mean')
        # Will plot only when max and min values are known
        pivot_list.append(pivot_df)
        vmin, vmax = min(vmin, pivot_df.min().min()), max(vmax, pivot_df.max().max())
        axs[i].set_title(f'{year}', fontsize=18)
        if i == 0:
            axs[i].set_ylabel(f'{index}', fontsize=20)
        if i == 2:
            axs[i].set_xlabel(f'{columns}', fontsize=20)
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
    [axs[i].imshow(pivot_df, cmap='gist_heat', vmin=vmin, vmax=vmax) for i, pivot_df in enumerate(pivot_list)]
    sm = ScalarMappable(cmap='gist_heat', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=axs)#.ravel().tolist())
    cb.ax.tick_params(labelsize=17)
    cb.ax.text(0.5, 1, f'Mean {values}', transform=cb.ax.transAxes,
               va='bottom', ha='center', fontsize=20)
    fig.suptitle(f'Yearly {values} Heatmaps per {index}, {columns}', fontsize=25)
    plt.show()
    plt.close()
    if similarity_heatmap:
        similarity_matrix = cosine_similarity([p.values.flatten() for p in pivot_list])
        return similarity_matrix


def similarity_matrix_plot(similarity_matrix, subject):
    mask = np.tril(np.ones_like(similarity_matrix, dtype=bool))
    ax = sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, mask=mask)
    plt.title(f'Cosine Similarity {subject}', fontsize=18)
    plt.xlabel('Relative Year', fontsize=14)
    plt.ylabel('Relative Year', fontsize=14)
    plt.show()
    plt.close()