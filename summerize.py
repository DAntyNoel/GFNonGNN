import os
import json
from tap import Tap

import matplotlib.pyplot as plt

def summarize_results(directory, type_='summary'):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    data = {}

    if type_ == 'full':
        all_keys = [
            "best_val_acc", "test_acc@best_val_acc", "epoch@best_val_ac",
            "time", "final_train_acc", "final_val_acc", "final_test_acc"
        ]
    elif type_ == 'summary':
        all_keys = [
            "best_val_acc", "test_acc@best_val_acc", "epoch@best_val_ac"
        ]
    else:
        raise ValueError("Invalid type. Use 'summary' or 'full'.")

    for folder in folders:
        folder_path = os.path.join(directory, folder)
        results_path = os.path.join(folder_path, 'results.json')
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                results = json.load(file)
                data[folder] = results

    rows = []
    for folder, results in data.items():
        row = [folder]
        for key in all_keys:
            row.append(results.get(key, 'N/A'))  
        rows.append(row)

    # 排序逻辑：根据 folder 名称中的 Data 部分排序
    def sort_key(folder):
        parts = folder.split('-')
        data_part = next((part for part in parts if part.startswith('Data')), '')
        return data_part, *parts # 其他部分按顺序排序

    rows.sort(key=lambda x: sort_key(x[0]))

    header = ['Settings'] + all_keys
    table = [header] + rows
    
    markdown_table = "| " + " | ".join(header) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in table[1:]:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
    
    latex_table = "\\begin{table}\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{" + "l" * len(header) + "}\n"
    latex_table += "\\hline\n"
    latex_table += " & ".join(header) + " \\\\ \\hline\n"
    for row in table[1:]:
        latex_table += " & ".join(map(str, row)) + " \\\\ \\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Summary of results}\n"
    latex_table += "\\end{table}\n"
    
    return markdown_table, latex_table


def plot_best_val_acc(directory):
    pic_dir = os.path.join(directory, 'pic')
    os.makedirs(pic_dir, exist_ok=True)

    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # {dataset: {config: {layer: best_val_acc}}}
    data = {}

    for folder in folders:
        folder_path = os.path.join(directory, folder)
        results_path = os.path.join(folder_path, 'results.json')

        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                results = json.load(file)
                best_val_acc = results.get("best_val_acc", None)
                if best_val_acc is not None:
                    parts:list = folder.split('-')
                    layer = next((part for part in parts if part.endswith('layer')), 'N/A')
                    dataset = next((part for part in parts if part.startswith('Data')), '??')
                    if layer in parts:
                        parts.remove(layer) 
                    if dataset in parts:
                        parts.remove(dataset)
                    layer = int(layer.replace('layer', ''))  
                    dataset = dataset.replace('Data_', '')
                    config = '-'.join(parts)  

                    if dataset not in data:
                        data[dataset] = {}
                    if config not in data[dataset]:
                        data[dataset][config] = {}
                    data[dataset][config][layer] = best_val_acc

    for dataset, configs in data.items():
        plt.figure(figsize=(10, 6))

        for config, layers in configs.items():
            sorted_layers = sorted(layers.items())
            layers, accuracies = zip(*sorted_layers)
            line_style = '--' if 'base' in config else '-'

            plt.plot(layers, accuracies, label=config, marker='o', linestyle=line_style)

        plt.xlabel('Layer')
        plt.ylabel('Best Validation Accuracy')
        plt.title(f'Best Validation Accuracy vs. Layer for {dataset}')
        plt.legend()
        plt.grid(True)

        pic_path = os.path.join(pic_dir, f'{dataset}.png')
        plt.savefig(pic_path)
        plt.close()

    print(f"Pictures saved to: {pic_dir}")


class Args(Tap):
    directory: str
    type_: str = 'summary'  # 'summary' or 'full'
    md: bool = False  # If True, save as markdown
    latex: bool = False  # If True, save as LaTeX
    plot: bool = True  # If True, plot the results

    def configure(self):
        self.add_argument('-d', '--directory', type=str, required=True, help='Project directory containing the result folders.')

if __name__ == '__main__':
    args = Args().parse_args()
    if args.plot:
        plot_best_val_acc(args.directory)

    if args.md or args.latex:
        markdown_table, latex_table = summarize_results(args.directory, args.type_)
        if args.md:
            print("Markdown Table:")
            print(markdown_table)
        if args.latex:
            print("LaTeX Table:")
            print(latex_table)