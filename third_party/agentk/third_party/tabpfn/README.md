# 🔍 TabPFN

### 📦 Creating the Environment

1. Create an environment named `tabpfn`.

    ```bash
    conda create -n tabpfn python=3.11 -y
    ```

2. Activate environment.

    ```bash
    conda activate tabpfn
    ```

3. Install dependency packages.
    ```bash
   pip install -r third_party/tabpfn/requirements.txt
   
   pip install git+https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/ramp-workflow.git@generative_regression_clean
   
   pip install git+https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/ramp-hyperopt.git@fe
    ```

## 🏁TabPFN

### 🛠 Install TabPFN from source code

```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git@98e9fc84346ce71dcb2b45bd73dffdc8dc999337"
```

### ▶️ Run TabPFN for a competition

 ```bash
python third_party/tabpfn/run_tabpfn.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
 ```

#### Example

```bash
python third_party/tabpfn/run_tabpfn.py --competition_name playground-series-s3e11 --workspace_path workspace_test --setup_path /path/to/setup/playground-series-s3e11/seed_0
```

You can specify the context window size using the --context_length option. The default value for TabPFN is 10,000.

 ```bash
python third_party/tabpfn/run_tabpfn.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
 ```

## 🏁TabPFN-extension

### 🛠 Install the specified version of tabpfn-extension from source code

```bash
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git@c49687fd30d9f7a2aa23260a9bcc2db79305381d"
```

### ▶️ Run TabPFN-extension for a competition

 ```bash
python third_party/tabpfn/run_tabpfn_extentions.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
 ```

You can specify the context window size using the --context_length option. The default value for TabPFN-extension is
10,000.

 ```bash
python third_party/tabpfn/run_tabpfn_extentions.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
```

## 🏁 TabICL

### 🛠 Install TabICL from source code

```bash
pip install git+https://github.com/soda-inria/tabicl.git@568edf8e1af29f3fe2c146cb550262e166edca83
```

### ▶️ Run TabICL for a competition

 ```bash
python third_party/tabpfn/run_tabicl.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
 ```

You can specify the context window size using the --context_length option. The default value for TabICL is 10,000.

 ```bash
python third_party/tabpfn/run_tabicl.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
```
