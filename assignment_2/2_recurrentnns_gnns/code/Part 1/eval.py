from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

def merge_csvs(base_path, csv_glob='*.csv'):
    all_csvs = glob(f'{base_path}/**/{csv_glob}')
    all_results = []

    for filename in all_csvs:
        df = pd.read_csv(filename)
        df['filename'] = filename
        all_results.append(df)
    
    merged_df = pd.concat(all_results, ignore_index=True)
    return merged_df
        
# Get all dataframes from all results
all_accs_df = merge_csvs('results', 'accs.csv')
all_loss_df = merge_csvs('results', 'losses.csv')

def plot_metric_for_model(df, model_name, metric, color, title, figname):
    df = df.query(f'model == "{model_name}"')
    metric_means = df.groupby('step').mean()[metric]
    metric_std = df.groupby('step').std()[metric]

    # Plotting starts
    plt.fill_between(range(len(metric_means)), metric_means - metric_std, metric_means + metric_std, alpha=0.5, color=color)
    plt.plot(range(len(metric_means)), metric_means, color=color)
    plt.title(title)
    plt.savefig(f'eval_results/{figname}')

    plt.cla()

plot_metric_for_model(all_loss_df, 'LSTM', 'loss', 'r', 'General LSTM Losses across seeds 42, 1337, 1994 every 10 steps', 'LSTM-plot-loss')
plot_metric_for_model(all_accs_df, 'LSTM', 'acc', 'g', 'General LSTM accuracies across seeds 42, 1337, 1994 every 10 steps', 'LSTM-plot-accs')
plot_metric_for_model(all_loss_df, 'peepLSTM', 'loss', 'r', 'General peepLSTM losses across seeds 42, 1337, 1994 every 10 steps', 'peepLSTM-plot-loss')
plot_metric_for_model(all_accs_df, 'peepLSTM', 'acc', 'g', 'General peepLSTM accuracies across seeds 42, 1337, 1994 every 10 steps', 'peepLSTM-plot-accs')