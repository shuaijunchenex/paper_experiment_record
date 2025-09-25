import os
import pandas as pd
import yaml

class ExpeimentHelper:
    def __init__(self):
        pass

    def helper_method(self):
        return "This is a helper method"
    
    @staticmethod
    def summarize_metrics_in_folder(folder, column, mode="max"):
        """
        扫描文件夹中的所有 CSV 文件，输出每个文件指定列的最大/最小值
        folder: 文件夹路径
        column: 列名，例如 'accuracy', 'average_loss'
        mode: 'max' 或 'min'
        """
        results = {}
        for fname in os.listdir(folder):
            if fname.endswith(".csv"):
                file_path = os.path.join(folder, fname)
                try:
                    df = pd.read_csv(file_path, header=1)  # 第 0 行是列名
                    if column not in df.columns:
                        print(f"[警告] {fname} 中没有列 {column}，跳过。")
                        continue
                    if mode == "max":
                        value = df[column].max()
                    elif mode == "min":
                        value = df[column].min()
                    else:
                        raise ValueError("mode 只能是 'max' 或 'min'")
                    results[fname] = value
                except Exception as e:
                    print(f"[错误] 读取 {fname} 失败: {e}")
        
        # 打印结果
        for k, v in results.items():
            print(f"文件名: {k} -> {column} {mode} = {v}")
        return results

    @staticmethod
    def _get_nested(config: dict, key: str, default="N/A"):
        """
        支持点号访问嵌套键，比如 general.device -> config["general"]["device"]
        """
        parts = key.split(".")
        val = config
        try:
            for p in parts:
                if isinstance(val, dict):
                    val = val[p]
                else:
                    return default
            return val
        except KeyError:
            return default

    @staticmethod
    def summarize_with_config(folder, column, mode="max", config_keys=None):
        """
        扫描文件夹中的所有 CSV 文件，输出每个文件的 config 键值和指定列的最大/最小值
        folder: 文件夹路径
        column: 列名，例如 'accuracy', 'average_loss'
        mode: 'max' 或 'min'
        config_keys: list[str]，支持点号路径，例如 ["dataset", "general.device", "client_selection.number"]
        """
        if config_keys is None:
            config_keys = []

        results = []
        for fname in os.listdir(folder):
            if fname.endswith(".csv"):
                file_path = os.path.join(folder, fname)
                # 读取第一行 Config
                with open(file_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip()
                    config_line = header.split("Config,", 1)[-1]

                # 尝试解析 YAML
                config = {}
                config = yaml.safe_load(config_line)

                # 读取数据部分（跳过 header）
                df = pd.read_csv(file_path, header=1)

                if mode == "max":
                    value = df[column].max()
                elif mode == "min":
                    value = df[column].min()

                # 提取多个 config 键值（支持嵌套）
                key_values = {k: ExpeimentHelper._get_nested(config, k, "N/A") for k in config_keys}
                results.append((fname, key_values, value))

        # 打印结果
        for fname, key_values, val in results:
            keys_str = " | ".join([f"{k}={v}" for k, v in key_values.items()])
            if keys_str:
                print(f"{fname} | {keys_str} | {column} {mode}={val}")
            else:
                print(f"{fname} | {column} {mode}={val}")

        return results