import os
import json
from tap import Tap

def summarize_results(directory, type_='summary'):
    # 获取目录下的所有文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # 初始化字典来存储数据
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

    
    # 遍历每个文件夹
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        results_path = os.path.join(folder_path, 'results.json')
        
        # 检查results.json是否存在
        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                results = json.load(file)
                data[folder] = results

    # 提取数据并排序
    rows = []
    for folder, results in data.items():
        row = [folder]
        for key in all_keys:
            row.append(results.get(key, 'N/A'))  # 如果某个key不存在，用'N/A'填充
        rows.append(row)
    # 排序逻辑：根据 folder 名称中的 Data 部分排序
    def sort_key(folder):
        parts = folder.split('-')
        data_part = next((part for part in parts if part.startswith('Data')), '')
        return data_part, *parts # 其他部分按顺序排序

    rows.sort(key=lambda x: sort_key(x[0]))

    # 构建表格
    # table = []
    header = ['Settings'] + all_keys
    table = [header] + rows
    
    # for folder, results in data.items():
    #     row = [folder]
    #     for key in all_keys:
    #         row.append(results.get(key, 'N/A'))  # 如果某个key不存在，用'N/A'填充
    #     table.append(row)
    
    # 输出Markdown格式
    markdown_table = "| " + " | ".join(header) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in table[1:]:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
    
    # 输出LaTeX格式
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

class Args(Tap):
    directory: str
    type_: str = 'summary'  # 'summary' or 'full'

    def configure(self):
        self.add_argument('-d', '--directory', type=str, required=True, help='Project directory containing the result folders.')

if __name__ == '__main__':
    args = Args().parse_args()
    markdown_table, latex_table = summarize_results(args.directory, args.type_)

    print("Markdown Table:")
    print(markdown_table)
    print("\nLaTeX Table:")
    print(latex_table)