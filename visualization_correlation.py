#%%
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import seaborn as sns

# %%

def plot(df_train, column1, column2, prefix='Raw'):
    plt.figure(figsize=(8, 6))  
    # plt.scatter(df_train[column1], df_train[column2], color='blue', alpha=0.6) 
    sns.regplot(x = df_train[column1], y = df_train[column2], color='green', scatter_kws={'alpha':0.5})
    plt.title(f'Scatter Plot of {column1} vs {column2} on {prefix}')  
    plt.xlabel(f'{column1}')  
    plt.ylabel(f'{column2}')  
    plt.grid(True) 
    plt.savefig(f'./{column1}_{column2}_{prefix}_correlation.png', dpi=300)

def test_significance(df_train, metric1, metric2):
    correlation, p_value = scipy.stats.spearmanr(df_train[metric1], df_train[metric2])
    print(f'Correlation between {metric1} and {metric2}')
    print(f"Correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")
    significant = False
    if p_value < 0.05:
        significant = True 
    return {
                    "metric1": metric1,
                    "metric2": metric2,
                    "correlation": correlation, 
                    "p-value": p_value, 
                    "signficant": significant
                }

def _get_polar(df):
    pos = np.exp(df.pos - df.ori)
    neg = np.exp(df.neg - df.ori)
    r = np.sqrt((pos ** 2 + neg ** 2))
    theta = np.arctan2(neg, pos)
    return np.stack([r, theta]).T


def normalize_column(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)
    return normalized_column

def generate_complete_df(df):
    api_score_path = './assets/sentiment_gradients_google_API_shuffle_concat.jsonl'
    api_score_df = pd.read_json(api_score_path, lines=True)
    api_score_df = api_score_df.rename({'input':'text'}, axis = 1)
    merged = pd.merge(df, api_score_df, on=['text', 'split'])
    X_train = _get_polar(merged)
    df_array = pd.DataFrame(X_train, columns=['radii', 'theta'])
    merged['radii'] = df_array['radii']
    merged['theta'] = df_array['theta']
    merged = merged.rename({'label_x':'label'}, axis = 1)
    return merged


if __name__ == '__main__':
    df = pd.read_csv("./assets/out.csv")
    df_train = df[df["split"] == "train"]

    X_train = _get_polar(df_train)
    df_array = pd.DataFrame(X_train, columns=['radii', 'theta'])
    df_train = pd.concat([df_train.reset_index(drop=True), df_array.reset_index(drop=True)], axis=1)
    df_train = df_train.dropna(subset=['theta', 'radii'])

    correlation_stats = []
    correlation_stats.append(test_significance(df_train, 'radii', 'api_magnitude')) # connotes how strongly sentimented the sentence is 
    correlation_stats.append(test_significance(df_train, 'theta', 'api_score')) # connotes how close we are to the x-axis i.e., the negative sentiment class
    correlation_stats.append(test_significance(df_train, 'label', 'theta')) # connotes how correlated our predictions are with the labels 
    correlation_stats.append(test_significance(df_train, 'label', 'api_score')) # connotes how correlated google-api predictions are with the labels 
    pd.DataFrame(correlation_stats).to_csv(f"./assets/correlation_stats.csv", index=False)

    plot(df_train, 'theta', 'api_score')
    plot(df_train, 'radii', 'api_magnitude')

    





# %%

# Plotting for shuffle and concatenated 
df = pd.read_csv("./assets/out_aug.csv")
df_shuffle = df[df["split"] == "shuffle"]
df_concat = df[df["split"] == "concat"]
prefix = ['shuffle', 'concat']
correlation_stats = []
for idx, df in enumerate([df_shuffle, df_concat]):
    pre = prefix[idx]
    df_train = generate_complete_df(df)
    print(df_train.isnull().sum()) 
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df_train = df_train.dropna(subset=['theta', 'radii', 'magnitude', 'score'])
    
    correlation_stats.append(test_significance(df_train, 'radii', 'magnitude')) # connotes how strongly sentimented the sentence is 
    correlation_stats.append(test_significance(df_train, 'theta', 'score')) # connotes how close we are to the x-axis i.e., the negative sentiment class
    correlation_stats.append(test_significance(df_train, 'label', 'theta')) # connotes how correlated our predictions are with the labels 
    correlation_stats.append(test_significance(df_train, 'label', 'score')) # connotes how correlated google-api predictions are with the labels 
    
    plot(df_train, f'theta', 'score', prefix=pre)
    plot(df_train, f'radii', 'magnitude', prefix=pre)
    breakpoint()

pd.DataFrame(correlation_stats).to_csv(f"./assets/correlation_shuffled_concat_stats.csv", index=False)



# %%
