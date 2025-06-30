import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_datasets():
    datasets = {}
    
    datasets['Vanilla LLM'] = pd.read_csv('data/pure_answers_temp01_results.csv')
    datasets['RAG+ LLM'] = pd.read_csv('data/rag_answers_temp01_results.csv')
    datasets['FT LLM'] = pd.read_csv('data/fine_answers_temp01_results.csv')
    datasets['Fine&RAG+ LLM'] = pd.read_csv('data/fine_rag_answers_temp01_results.csv')
    
    return datasets

def plot_style():
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white'
    })

def av_f1(datasets):
    avg_f1 = {}
    
    for name, df in datasets.items():
        avg_f1[name] = df['f1'].mean()
    
    return avg_f1

def bar_plot(avg_f1):
    colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']
    dataset_names = list(avg_f1.keys())
    values = list(avg_f1.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(dataset_names, values, color=colors)
    ax.set_xlabel('F1')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 0.01, i, f'{value:.3f}', 
                va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def box_plot(datasets):
    colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']
    
    data_to_plot = []
    labels = []
    
    for name, df in datasets.items():
        data_to_plot.append(df['f1'].values)
        labels.append(name)
    
    plt.figure(figsize=(10, 6))
    
    bp1 = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                      medianprops={'color': 'navy', 'linewidth': 1})
    
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('navy')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def main():
    datasets = load_datasets()
    plot_style()
    avg_f1 = av_f1(datasets)
    for name, f1_score in avg_f1.items():
        print(f"{name}: {f1_score:.3f}")
    print()
    bar_plot(avg_f1)
    box_plot(datasets)

if __name__ == "__main__":
    main()