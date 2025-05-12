import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_results(csv_file, output_image, mode):
    try:
        df = pd.read_csv(csv_file)
        
        # Переименовываем столбцы для унификации
        df = df.rename(columns={'param': 'Threads' if mode != 'cuda' else 'Block Size',
                               'time': 'Multiplication Time (s)'})
        
        df['Time (ms)'] = df['Multiplication Time (s)'] * 1000
        
        if mode == 'cuda':
            # Обработка данных для CUDA
            dimensions = sorted(df['size'].unique(), reverse=True)
            blocks = sorted(df['Threads'].unique())
            
            time_pivot = df.pivot_table(
                index='size',
                columns='Threads',
                values='Time (ms)',
                aggfunc='median'
            ).reindex(index=dimensions, columns=blocks)
            
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(
                time_pivot,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu_r",
                linewidths=0.5,
                cbar_kws={'label': 'Время (мс)'}
            )
            
            plt.title('Производительность умножения матриц (CUDA)', fontsize=16, pad=20)
            plt.xlabel('Размер блока', fontsize=12)
            plt.ylabel('Размер матрицы', fontsize=12)
            
        else:
            # Обработка данных для других режимов
            dimensions = sorted(df['size'].unique(), reverse=True)
            threads = sorted(df['Threads'].unique())
            
            time_pivot = df.pivot_table(
                index='size',
                columns='Threads',
                values='Time (ms)',
                aggfunc='median'
            ).reindex(index=dimensions, columns=threads)
            
            plt.figure(figsize=(16, 10))
            ax = sns.heatmap(
                time_pivot,
                annot=True,
                fmt="d",
                cmap="YlGnBu_r",
                linewidths=0.5,
                cbar_kws={'label': 'Время (мс)'}
            )
            
            best_dim = time_pivot.stack().idxmin()[0]
            best_threads = time_pivot.stack().idxmin()[1]
            best_time = time_pivot.at[best_dim, best_threads]
            
            plt.title(f'Производительность умножения матриц ({mode.capitalize()})', fontsize=16, pad=20)
            plt.xlabel('Потоки' if mode == 'parallel' else 'Процессы', fontsize=12)
            plt.ylabel('Размер матрицы', fontsize=12)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='heatmap.png')
    parser.add_argument('--mode', default='parallel', choices=['serial', 'parallel', 'cuda', 'mpi'])
    args = parser.parse_args()
    
    plot_results(args.csv, args.output, args.mode)