import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ast
import os
import requests
import zipfile
import io
import tempfile
import shutil
import re # Import regex for URL parsing

MODEL_FILENAME = 'random_forest_energy_model.joblib'
FEATURES_ORDER = ['LOC', 'No_of_Functions', 'No_of_Classes', 'No_of_Loops',
                  'Loop_Nesting_Depth', 'No_of_Conditional_Blocks', 'Import_Score',
                  'I/O Calls']

library_weights = {
    "torch": 10, "tensorflow": 10, "jax": 9, "keras": 9, "transformers": 9, "lightgbm": 8,
    "xgboost": 8, "catboost": 8, "sklearn": 7, "scikit-learn": 7, "pandas": 6, "numpy": 6,
    "dask": 7, "polars": 6, "matplotlib": 4, "seaborn": 4, "plotly": 5, "bokeh": 5,
    "altair": 4, "cv2": 6, "PIL": 4, "imageio": 3, "scikit-image": 4, "nltk": 5, "spacy": 6,
    "gensim": 5, "requests": 2, "httpx": 2, "urllib": 1, "aiohttp": 3, "fastapi": 4, "flask": 3,
    "django": 5, "openpyxl": 3, "csv": 1, "json": 1, "sqlite3": 2, "sqlalchemy": 3, "h5py": 4,
    "pickle": 2, "os": 1, "sys": 1, "shutil": 1, "glob": 1, "pathlib": 1, "math": 1,
    "statistics": 1, "scipy": 5, "datetime": 1, "time": 1, "calendar": 1, "re": 1,
    "argparse": 1, "typing": 1, "logging": 1, "threading": 2, "multiprocessing": 3,
    "concurrent": 3, "subprocess": 3, "random": 1, "uuid": 1, "hashlib": 1, "base64": 1,
    "decimal": 1, "boto3": 6, "google.cloud": 6, "azure": 6, "pyspark": 9, "IPython": 2,
    "jupyter": 2
}
file_io_funcs = {'open', 'read', 'write', 'remove', 'rename', 'copy', 'seek', 'tell', 'flush'}

class FeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.max_loop_depth = 0; self.current_depth = 0; self.file_io_calls = 0
    def visit_For(self, node):
        self.current_depth += 1; self.max_loop_depth = max(self.max_loop_depth, self.current_depth)
        self.generic_visit(node); self.current_depth -= 1
    def visit_While(self, node):
        self.current_depth += 1; self.max_loop_depth = max(self.max_loop_depth, self.current_depth)
        self.generic_visit(node); self.current_depth -= 1
    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name): func_name = node.func.id
        elif isinstance(node.func, ast.Attribute): func_name = node.func.attr
        if func_name in file_io_funcs: self.file_io_calls += 1
        self.generic_visit(node)

def extract_features_from_file(file_path):

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f: source = f.read()
    except Exception as e:
        print(f"Error reading script file {file_path}: {e}") # Log error
        st.error(f"Could not read file: {os.path.basename(file_path)}") # Show error in UI
        return None
    try:
        tree = ast.parse(source)
    except Exception as e:
         print(f"Error parsing script {file_path} with AST: {e}") # Log error
         st.error(f"Could not parse file (invalid Python?): {os.path.basename(file_path)}") # Show error in UI
         return None

    num_lines = len(source.splitlines())
    num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    num_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    num_loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    num_conditional_blocks = sum(isinstance(node, ast.If) for node in ast.walk(tree))
    num_imports = 0; imported_libs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                lib = name.name.split('.')[0];
                if lib: imported_libs.add(lib)
                num_imports += 1
        elif isinstance(node, ast.ImportFrom):
             if node.module: lib = node.module.split('.')[0];
             if lib: imported_libs.add(lib)
             num_imports += 1
    weighted_import_score = sum(library_weights.get(lib, 2) for lib in imported_libs)
    extractor = FeatureExtractor(); extractor.visit(tree)
    features_dict = {
        'LOC': num_lines, 'No_of_Functions': num_functions, 'No_of_Classes': num_classes,
        'No_of_Loops': num_loops, 'Loop_Nesting_Depth': extractor.max_loop_depth,
        'No_of_Conditional_Blocks': num_conditional_blocks, 'Import_Score': weighted_import_score,
        'I/O Calls': extractor.file_io_calls
    }
    return features_dict # Just return the dictionary

def predict_for_features(model, features_dict):
    
    try:
        input_features = {key: [pd.to_numeric(value, errors='coerce')] for key, value in features_dict.items()}
        input_df = pd.DataFrame(input_features, columns=FEATURES_ORDER)
        if input_df.isnull().values.any():
             st.error("Error: Non-numeric values found in extracted features during prediction step.")
             return None
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction step: {e}")
        return None


st.set_page_config(layout="wide")
st.title("🐍 Sustainable Code Reviewer (Prototype)")

# --- Load Model ---
@st.cache_resource # Cache the loaded model
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file '{filename}' not found. Ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load model '{filename}'. Error: {e}")
        st.stop()

loaded_model = load_model(MODEL_FILENAME)
st.success(f"Model loaded successfully.")

# --- User Input ---
st.header("Input")
input_path_or_url = st.text_input(
    "Enter local path to Python script/directory OR public GitHub repository URL:",
    placeholder="e.g., C:\\path\\to\\script.py or https://github.com/user/repo/tree/branch/subdir"
)

analyze_button = st.button("Analyze and Predict")

st.header("Results")

if analyze_button and input_path_or_url:
    files_to_process = []
    scan_source_description = ""
    temp_dir_context = None
    extracted_repo_root = None # Path inside temp_dir holding repo files
    base_path_for_relative = None # Base path for relative display (local or extracted)
    target_subdir_in_repo = None # Subdir specified in URL

    with st.spinner("Accessing source and finding Python files..."):
        # Check if input is a GitHub URL
        if input_path_or_url.startswith(('http://', 'https://')) and 'github.com' in input_path_or_url:
            scan_source_description = f"GitHub source: {input_path_or_url}"
            st.info(f"Processing {scan_source_description}")

            # --- Improved URL Parsing ---
            match = re.match(r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(/(.*))?)?", input_path_or_url)
            if not match:
                st.error("Could not parse GitHub URL structure. Please provide URL to repo root or subdirectory.")
                st.stop()
            user, repo, branch, _, subdir = match.groups()
            repo = repo.replace(".git", "")
            target_subdir_in_repo = subdir.strip('/') if subdir else None
            repo_url_base = f"https://github.com/{user}/{repo}"
            st.write(f"Detected Repository Base: {repo_url_base}")
            if target_subdir_in_repo: st.write(f"Detected Target Subdirectory: {target_subdir_in_repo}")
            # --- End Improved URL Parsing ---

            # Attempt download using common branches (use specified branch first if available)
            potential_branches = [branch] if branch else ['main', 'master']
            repo_zip_content = None
            for b in potential_branches:
                potential_zip_url = f"{repo_url_base}/archive/refs/heads/{b}.zip"
                st.write(f"Attempting to download zip from branch: {b}...")
                try:
                    response = requests.get(potential_zip_url, stream=True, timeout=30)
                    response.raise_for_status()
                    repo_zip_content = response.content
                    st.write(f"Successfully downloaded zip for branch: {b}")
                    break
                except requests.exceptions.RequestException as e:
                     st.write(f"Could not download zip for branch '{b}': {e}")
                except Exception as e:
                     st.write(f"An unexpected error occurred downloading branch '{b}': {e}")

            if not repo_zip_content:
                st.error("Could not download repository zip. Check URL or repo structure (e.g., branch name).")
                st.stop()

            # Extract to temporary directory
            try:
                temp_dir_context = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_context.name
                st.write(f"Extracting repository to temporary location...")
                with zipfile.ZipFile(io.BytesIO(repo_zip_content)) as zf:
                    zf.extractall(temp_dir)
                    zip_root_folder = zf.namelist()[0].split('/')[0] # e.g., 'repo-main'
                    extracted_repo_root = os.path.join(temp_dir, zip_root_folder)
                st.write("Extraction complete.")

                # Determine the starting path for os.walk
                scan_start_path = extracted_repo_root
                base_path_for_relative = extracted_repo_root # For display
                if target_subdir_in_repo:
                    potential_subdir_path = os.path.join(extracted_repo_root, target_subdir_in_repo.replace('%20', ' '))
                    if os.path.isdir(potential_subdir_path):
                         scan_start_path = potential_subdir_path
                         base_path_for_relative = scan_start_path # Make path relative to subdir
                         st.write(f"Scanning specifically within subdirectory: {target_subdir_in_repo}")
                    else:
                         st.warning(f"Subdirectory '{target_subdir_in_repo}' not found in extracted repo. Scanning entire repository.")

                # Find Python files starting from the scan_start_path
                for root, dirs, files in os.walk(scan_start_path):
                    dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', '.git']]
                    for file in files:
                        if file.endswith(".py"): files_to_process.append(os.path.join(root, file))

            except Exception as e:
                st.error(f"Error during zip extraction or file scanning: {e}")
                if temp_dir_context: temp_dir_context.cleanup()
                st.stop()

        # Check if input is a local path
        elif os.path.exists(input_path_or_url):
            base_path_for_relative = input_path_or_url # Store base path
            if os.path.isfile(input_path_or_url) and input_path_or_url.endswith(".py"):
                scan_source_description = f"local file: {input_path_or_url}"
                st.info(f"Processing {scan_source_description}")
                files_to_process.append(input_path_or_url)
                base_path_for_relative = os.path.dirname(input_path_or_url) # Base for relative path is dir
            elif os.path.isdir(input_path_or_url):
                scan_source_description = f"local directory: {input_path_or_url}"
                st.info(f"Processing {scan_source_description}")
                for root, dirs, files in os.walk(input_path_or_url):
                    dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', '.git']]
                    for file in files:
                        if file.endswith(".py"): files_to_process.append(os.path.join(root, file))
            else:
                st.error(f"Local path exists but is not a directory or a .py file: {input_path_or_url}")
                st.stop()
        else:
            st.error(f"Input path or URL not found or not recognized: {input_path_or_url}")
            st.stop()
        # --- End Determine Input Type ---

    # --- Process Files ---
    if not files_to_process:
        st.warning("No Python files found to process in the specified location.")
    else:
        st.info(f"Found {len(files_to_process)} Python file(s). Analyzing...")
        results_list = []
        overall_success_count = 0
        results_placeholder = st.container() # Use a container to group results

        with results_placeholder:
            # Don't use a spinner here, show results progressively
            for file_path in files_to_process:
                # Determine relative path for display
                display_path = os.path.basename(file_path) # Default
                try:
                     if base_path_for_relative and os.path.commonpath([base_path_for_relative, file_path]) == os.path.normpath(base_path_for_relative):
                          display_path = os.path.relpath(file_path, base_path_for_relative)
                except ValueError: # Handles different drive letters etc.
                     display_path = os.path.basename(file_path) # Fallback

                st.subheader(f"Results for: {display_path}") # Display filename first

                features_dict = extract_features_from_file(file_path) # Function now handles errors internally via st.error

                if features_dict:
                    # Display features
                    st.write("🔍 **Extracted Features:**")
                    output_str = ""
                    for key, value in features_dict.items():
                        output_str += f"  • {key.replace('_', ' '):<25} : {value}\n"
                    st.code(output_str, language=None)

                    # Predict
                    prediction = predict_for_features(loaded_model, features_dict)

                    if prediction is not None:
                        st.success(f"**Predicted Energy: {prediction:.2f} joules**")
                        results_list.append({'file': display_path, 'predicted_joules': prediction})
                        overall_success_count += 1
                    # Error message for prediction failure is handled inside predict_for_features
                # Error message for feature extraction failure is handled inside extract_features_from_file

                st.divider() # Add divider between files

            st.info(f"Processing finished. Successfully predicted for {overall_success_count} out of {len(files_to_process)} Python file(s) processed.")

    # Cleanup temporary directory if it was created
    if temp_dir_context:
        try:
            temp_dir_context.cleanup()
            st.write("Temporary directory cleaned up.")
        except Exception as e:
            st.warning(f"Could not automatically clean up temp directory. Error: {e}")

elif analyze_button:
    st.warning("Please enter a local path or a GitHub URL.")

