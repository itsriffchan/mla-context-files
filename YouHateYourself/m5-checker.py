import json
import re
import sys
import os

def check_notebook(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.ipynb'):
                notebook = json.load(f)
                code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
                code_content = "\n".join(["".join(cell.get('source', [])) for cell in code_cells])
            else:
                code_content = f.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Normalize whitespace aggressively
    normalized_code = re.sub(r'\s+', ' ', code_content)
    
    # Ultra-lenient regexes
    checks = {
        "Manual Search Setup": [
            (r"n_estimators_options\s*=\s*\[\s*50\s*,\s*100\s*,\s*200\s*\]", "n_estimators_options = [50, 100, 200]"),
            (r"max_depth_options\s*=\s*\[\s*None\s*,\s*10\s*,\s*20\s*,\s*30\s*\]", "max_depth_options = [None, 10, 20, 30]"),
            (r"min_samples_split_options\s*=\s*\[\s*2\s*,\s*5\s*,\s*10\s*\]", "min_samples_split_options = [2, 5, 10]")
        ],
        "Grid Search Execution": [
            (r"GridSearchCV\s*\(\s*estimator\s*=\s*model,\s*param_grid\s*=\s*param_grid,\s*cv\s*=\s*5,\s*n_jobs\s*=\s*-1,\s*scoring\s*=\s*'accuracy'\s*\)", "GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')")
        ],
        "Randomized Search Execution": [
            (r"RandomizedSearchCV\s*\(", "RandomizedSearchCV(...)")
        ],
        "Bayesian Optimization Execution": [
            (r"param_bounds\s*=\s*\{\s*'n_estimators'\s*:\s*\(\s*50\s*,\s*200\s*\)", "param_bounds definitions"),
            (r"optimizer\.maximize\s*\(\s*init_points\s*=\s*5\s*,\s*n_iter\s*=\s*25\s*\)", "optimizer.maximize(init_points=5, n_iter=25)")
        ],
        "Data Splitting (Train/Test/Val)": [
            (r"train_test_split\s*\(.*?test_size\s*=\s*0\.3.*?random_state\s*=\s*42\s*\)", "train_test_split(..., test_size=0.3, random_state=42)"),
            (r"train_test_split\s*\(.*?test_size\s*=\s*0\.2\D.*?random_state\s*=\s*42\s*\)", "train_test_split(..., test_size=0.2, random_state=42)"),
            (r"train_test_split\s*\(.*?test_size\s*=\s*0\.25.*?random_state\s*=\s*42\s*\)", "train_test_split(..., test_size=0.25, random_state=42)")
        ],
        "Cross-Validation Methods": [
            (r"KFold\s*\(\s*n_splits\s*=\s*5,\s*shuffle\s*=\s*True,\s*random_state\s*=\s*42\s*\)", "KFold(n_splits=5, shuffle=True, random_state=42)"),
            (r"StratifiedKFold\s*\(", "StratifiedKFold(...)"),
            (r"LeaveOneOut\s*\(\s*\)", "LeaveOneOut()"),
            (r"ShuffleSplit\s*\(", "ShuffleSplit(...)")
        ],
        "Classification Metrics": [
            (r"precision_score\s*\(.*?average\s*=\s*'macro'\s*\)", "precision_score(..., average='macro')"),
            (r"recall_score\s*\(.*?average\s*=\s*'macro'\s*\)", "recall_score(..., average='macro')"),
            (r"f1_score\s*\(.*?average\s*=\s*'macro'\s*\)", "f1_score(..., average='macro')"),
            (r"confusion_matrix\s*\(\s*y_true\s*,\s*y_pred\s*\)", "confusion_matrix(y_true, y_pred)"),
            (r"roc_auc_score\s*\(", "roc_auc_score()")
        ],
        "Regression Metrics": [
            (r"mean_absolute_error\s*\(", "mean_absolute_error()"),
            (r"mean_squared_error\s*\(", "mean_squared_error()"),
            (r"np\.sqrt\s*\(\s*mean_squared_error", "np.sqrt(mean_squared_error(...))"),
            (r"r2_score\s*\(", "r2_score()")
        ]
    }

    print(f"\n🔍 Validating '{filepath}' against m5_model_evaluation_and_improvement concepts...\n")
    
    total_passed = 0
    total_checks = sum(len(v) for v in checks.values())

    for section, patterns in checks.items():
        print(f"--- {section} ---")
        for pattern, desc in patterns:
            # Check against normalized code
            if re.search(pattern, normalized_code) or re.search(pattern, code_content):
                print(f"  [✓] PASSED: {desc}")
                total_passed += 1
            else:
                print(f"  [x] FAILED: Missing or incorrect {desc}")
        print()

    print(f"\n📊 Final Score: {total_passed}/{total_checks} checks passed.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python m5_checker.py <your_script.py_or_notebook.ipynb>")
    else:
        check_notebook(sys.argv[1])
