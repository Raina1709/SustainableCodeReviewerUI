# streamlit_app.py (v7 - Added Rough Estimated Savings)
# UI for Python Script Energy Consumption Prediction + Recommendations

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
import re
from openai import AzureOpenAI # Import Azure OpenAI client

# --- Configuration ---
MODEL_FILENAME = 'random_forest_energy_model.joblib'
FEATURES_ORDER = ['LOC', 'No_of_Functions', 'No_of_Classes', 'No_of_Loops',
                    'Loop_Nesting_Depth', 'No_of_Conditional_Blocks', 'Import_Score',
                    'I/O Calls']

# --- Feature Extraction Code ---
# (Keep the FeatureExtractor and extract_features_and_code_from_file functions as they are)
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
def extract_features_and_code_from_file(file_path):
    source_code = None
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading script file {file_path}: {e}")
        st.error(f"Could not read file: {os.path.basename(file_path)}")
        return None, None
    try:
        tree = ast.parse(source_code)
    except Exception as e:
        print(f"Error parsing script {file_path} with AST: {e}")
        st.error(f"Could not parse file (invalid Python?): {os.path.basename(file_path)}")
        return None, source_code
    num_lines = len(source_code.splitlines())
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
    return features_dict, source_code
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

def get_openai_recommendations(source_code, features_dict):
    """
    Sends code and features to Azure OpenAI for recommendations.
    Tries to extract a "Total Estimated Energy Savings: X-Y%" first.
    If not found, falls back to summing individual "(Estimated Saving: X-Y%)".
    """
    recommendations = "Could not retrieve recommendations." # Default message
    potential_savings_percent = 0.0  # Initialize the return value

    # Helper function to parse and sum matched percentage strings
    def _parse_and_sum_savings_matches(matches_list):
        current_sum = 0.0
        if matches_list:
            for match_tuple in matches_list:
                try:
                    primary_value_str = match_tuple[0]    # X from "X%" or "X-Y%"
                    secondary_value_str = match_tuple[1]  # Y from "X-Y%" (can be empty)
                    
                    value_to_add_str = ""
                    if secondary_value_str and secondary_value_str.strip(): # Y value from X-Y% range
                        value_to_add_str = secondary_value_str.strip()
                    elif primary_value_str and primary_value_str.strip(): # X value
                        value_to_add_str = primary_value_str.strip()
                    
                    if value_to_add_str:
                        current_sum += float(value_to_add_str)
                except ValueError:
                    # Optional: st.warning(f"Could not parse number from saving value: {match_tuple}")
                    pass # Skip if parsing fails for this specific match_tuple
        return current_sum

    try:
        # --- IMPORTANT: Configure Credentials ---
        if not all(k in st.secrets for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]):
            st.error("Azure OpenAI credentials missing in Streamlit Secrets (secrets.toml).")
            # ... (rest of credential error messages)
            return "Credentials configuration missing.", potential_savings_percent
        # ... (rest of credential loading and checks) ...
        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
        deployment_name = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

        if not all([api_key, azure_endpoint, api_version, deployment_name]):
            st.error("One or more Azure OpenAI credentials in Streamlit Secrets are empty.")
            return "Credentials configuration incomplete.", potential_savings_percent

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )

        # --- Construct the Prompt ---
        features_str = "\n".join([f"- {key.replace('_', ' ')}: {value}" for key, value in features_dict.items()])
        prompt_messages = [
            {"role": "system", "content": "You are an expert Python programmer specialized in code optimization for energy efficiency. Analyze the provided code and its static features to suggest specific, actionable changes that would likely reduce its energy consumption during execution."},
            {"role": "user", "content": f"""Please analyze the following Python code for potential energy optimizations.

            Consider factors like:
            - Algorithmic efficiency, loop optimizations, I/O operations, library usage, concurrency, memory usage.

            Provide specific, actionable recommendations.
            If possible, for each recommendation, include an estimated percentage improvement using the exact format: **(Estimated Saving: X%)** or **(Estimated Saving: X-Y%)**.
            Alternatively, or in addition, you can provide an overall summary figure like **Total Estimated Energy Savings: Z%** or **Total Estimated Energy Savings: Y-Z%**.

            If providing specific savings per item AND an overall total, ensure the overall total is the definitive figure.

            Source Code:
            ```python
            {source_code}
            ```
            Code Features:
            {features_str}

            Recommendations:
            """} # Prompt updated to acknowledge both formats
        ]

        # --- Make API Call ---
        st.write("_Contacting Azure OpenAI... (This may take a moment)_")
        response = client.chat.completions.create(
            model=deployment_name,
            messages=prompt_messages,
            temperature=0.5,
            max_tokens=4000,
            # ... (other API parameters)
        )

        # --- Extract Response and Attempt to Parse Savings ---
        if response.choices:
            recommendations = response.choices[0].message.content.strip()

            # --- DEBUGGING: Uncomment to see the raw AI response ---
            # st.markdown("--- AI Raw Response for Debugging ---")
            # st.text_area("Raw Response:", recommendations, height=200)
            # st.markdown("--- End AI Raw Response ---")

            if not recommendations:
                recommendations = "AI model returned an empty recommendation."
            else:
                # Regex for "Total Estimated Energy Savings: X-Y%" or "Total Estimated Energy Saving: X%"
                # Captures X (group 1) and Y (group 2, optional)
                regex_overall_savings = r"Total Estimated Energy(?: Savings?)?:\s*(\d+\.?\d*)\s*-?\s*(\d*\.?\d*)?%?"
                
                # Regex for per-item "(Estimated Saving: X-Y%)"
                # Captures X (group 1) and Y (group 2, optional)
                regex_itemized_savings = r"\(\s*Estimated Saving:\s*(\d+\.?\d*)\s*-?\s*(\d*\.?\d*)?%?\s*\)"

                overall_savings_matches = re.findall(regex_overall_savings, recommendations, re.IGNORECASE)

                if overall_savings_matches:
                    # If "Total Estimated Energy Savings" is found, prioritize it.
                    # Typically, there should be one such statement, but _parse_and_sum_savings_matches will sum if there are multiple.
                    potential_savings_percent = _parse_and_sum_savings_matches(overall_savings_matches)
                    # st.info(f"Found and used 'Total Estimated Energy Savings': {potential_savings_percent:.2f}%") # Debug
                else:
                    # If no "Total..." line is found, fall back to summing individual itemized savings
                    itemized_savings_matches = re.findall(regex_itemized_savings, recommendations, re.IGNORECASE)
                    if itemized_savings_matches:
                        potential_savings_percent = _parse_and_sum_savings_matches(itemized_savings_matches)
                        # st.info(f"Used sum of itemized '(Estimated Saving)': {potential_savings_percent:.2f}%") # Debug
                    # If neither is found, potential_savings_percent remains 0.0
        else:
            recommendations = "No recommendations received from API (response structure unexpected)."

    except ImportError:
        st.error("The 'openai' library is not installed...") # Abridged for brevity
        recommendations = "OpenAI library not found."
    except KeyError as e:
        st.error(f"Azure OpenAI credential '{e}' not found...") # Abridged
        recommendations = f"Credential configuration missing: {e}"
    except Exception as e:
        st.error(f"Error calling Azure OpenAI API: {e}")
        recommendations = f"Error fetching recommendations: {e}"
        # import traceback
        # st.text(traceback.format_exc()) # For detailed error

    return recommendations, potential_savings_percent

#--- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Sustainable Code Review Assistant")
st.title("🐍 Sustainable Code Review Assistant")

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

#--- Load Model ---
loaded_model = load_model(MODEL_FILENAME)
st.success(f"Prediction Model loaded successfully.")

# --- State Variables for Summary ---
if 'total_scripts_analyzed' not in st.session_state:
    st.session_state['total_scripts_analyzed'] = 0
if 'total_predicted_consumption' not in st.session_state:
    st.session_state['total_predicted_consumption'] = 0
if 'potential_total_savings' not in st.session_state:
    st.session_state['potential_total_savings'] = 0

# --- Tabs ---
tab1, tab2 = st.tabs(["Analyze Code", "Summary"])

# --- Analyze Code Tab ---
with tab1:
    st.header("Input")
    input_path_or_url = st.text_input(
        "Enter public GitHub repository URL or local file/directory path:",
        placeholder="https://github.com/skills/introduction-to-github"
    )

    analyze_button = st.button("Analyze and Predict")

    # --- Analysis and Prediction Output Area ---
    files_to_process = []
    scan_source_description = ""
    temp_dir_context = None
    extracted_repo_root = None
    base_path_for_relative = None
    target_subdir_in_repo = None

    # --- Determine Input Type and Get Files ---
    if analyze_button:
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

        elif os.path.exists(input_path_or_url):
            with st.spinner("Accessing source and finding Python files..."):
                base_path_for_relative = input_path_or_url # Store base path
                if os.path.isfile(input_path_or_url) and input_path_or_url.endswith(".py"):
                    scan_source_description = f"local file: {input_path_or_url}"
                    st.info(f"Processing {scan_source_description}")
                    files_to_process.append(input_path_or_url)
                    base_path_for_relative = os.path.dirname(input_path_or_url) # Use dir for relative path
                elif os.path.isdir(input_path_or_url):
                    scan_source_description = f"local directory: {input_path_or_url}"
                    st.info(f"Processing {scan_source_description}")
                    for root, dirs, files in os.walk(input_path_or_url):
                        dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', '.git']]
                        for file in files:
                            if file.endswith(".py"): files_to_process.append(os.path.join(root, file))
                else:
                    st.error(f"Local path is not a directory or a .py file: {input_path_or_url}")
        else:
            st.error(f"Input path or URL not found or not recognized: {input_path_or_url}")

        # --- Process Files ---
        st.header("Analysis Results")
        results_placeholder = st.container() # Use a container to group results
        if not files_to_process:
            st.warning("No Python files found to process.")
        else:
            total_predicted_energy = 0
            total_potential_savings = 0
            total_scripts = 0

            with results_placeholder:
                for file_path in files_to_process:
                    display_path = os.path.basename(file_path) # Default
                    try: # Try getting relative path
                        if base_path_for_relative and os.path.commonpath([base_path_for_relative, file_path]) == os.path.normpath(base_path_for_relative):
                            display_path = os.path.relpath(file_path, base_path_for_relative)
                    except ValueError: display_path = os.path.basename(file_path)

                    st.subheader(f"Results for: {display_path}")

                    # Extract features AND source code
                    features_dict, source_code = extract_features_and_code_from_file(file_path) # Handles its own errors via st.error

                    if features_dict and source_code:
                        # Display features
                        st.write("🔍 **Extracted Features:**")
                        output_str = "\n".join([f"  • {key.replace('_', ' '):<25} : {value}" for key, value in features_dict.items()])
                        st.code(output_str, language=None)

                        # Predict Energy
                        prediction = predict_for_features(loaded_model, features_dict) # Handles its own errors via st.error

                        if prediction is not None:
                            st.success(f"**Predicted Energy: {prediction:.2f} joules**")
                            total_predicted_energy += prediction
                            total_scripts += 1

                            # Get OpenAI Recommendations (Modified to return savings percent)
                            st.write("💡 **Fetching Energy Saving Recommendations...**")
                            with st.spinner("Contacting Azure OpenAI..."):
                                recommendations, savings_percent = get_openai_recommendations(source_code, features_dict)
                            st.markdown("**Recommendations:**")
                            st.markdown(recommendations) # Display recommendations using markdown

                            if savings_percent > 0:
                                estimated_saving = prediction * (savings_percent / 100.0)
                                st.info(f"**Estimated Potential Saving:** {estimated_saving:.2f} joules (based on AI recommendation of up to {savings_percent:.0f}% improvement)")
                                total_potential_savings += estimated_saving

                    st.divider() # Add divider between files

            # Update session state for summary tab
            st.session_state['total_scripts_analyzed'] = total_scripts
            st.session_state['total_predicted_consumption'] = total_predicted_energy
            st.session_state['potential_total_savings'] = total_potential_savings

            if total_scripts > 0:
                st.info(f"Analysis completed for {total_scripts} Python scripts.")
            else:
                st.info("No Python scripts were analyzed.")

        # Cleanup temporary directory
        if temp_dir_context:
            try:
                temp_dir_context.cleanup()
                st.write("Temporary directory cleaned up.")
            except Exception as e:
                st.warning(f"Could not automatically clean up temp directory. Error: {e}")

# --- Summary Tab ---
with tab2:
    st.header("Analysis Summary")

    # Retrieve values from session state
    total_scripts_analyzed = st.session_state.get('total_scripts_analyzed', 0)
    total_predicted_consumption_single_run = st.session_state.get('total_predicted_consumption', 0.0)
    potential_total_savings_single_run = st.session_state.get('potential_total_savings', 0.0)

    # Display the first three original metrics
    st.metric("Total Scripts Analyzed", total_scripts_analyzed)
    st.metric("Total Predicted Energy Consumption (per single combined execution)", f"{total_predicted_consumption_single_run:.2f} joules")
    st.metric("Estimated Total Potential Saving (per single combined execution)", f"{potential_total_savings_single_run:.2f} joules")

    # Calculate Improvement in Efficiency based on single run
    improvement_percentage_single_run = 0.0
    if total_predicted_consumption_single_run > 0:
        improvement_percentage_single_run = (potential_total_savings_single_run / total_predicted_consumption_single_run) * 100
    st.metric("Improvement in Efficiency (for single execution)", f"{improvement_percentage_single_run:.1f}%")
    st.markdown(f"""
    <div style="margin-top: 1rem; margin-bottom: 0.5rem;"> <div style="font-size: 0.875rem; color: rgb(85, 87, 97); line-height: 1.25rem;">{label_improvement}</div>
        <div style="font-size: 1.875rem; font-weight: 600; color: green; line-height: 2.25rem;">{value_improvement}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider() # Add a visual separator

    st.subheader("Projected Daily Impact (Based on 500 executions/day)")

    if potential_total_savings_single_run > 0:
        # --- New Calculation: Estimated savings on multiple executions ---
        EXECUTIONS_PER_DAY = 500
        estimated_daily_savings_joules = potential_total_savings_single_run * EXECUTIONS_PER_DAY
        st.metric("Estimated Daily Savings (on 500 executions)", f"{estimated_daily_savings_joules:,.2f} joules")

        # --- New Calculation: Bulb-hours equivalent ---
        ENERGY_PER_12W_CFL_HOUR_JOULES = 43200.0  # Energy for one 12W CFL bulb in 1 hour

        if ENERGY_PER_12W_CFL_HOUR_JOULES > 0:
            # Total bulb-hours this daily saving represents
            total_bulb_hours_equivalent = estimated_daily_savings_joules / ENERGY_PER_12W_CFL_HOUR_JOULES

            st.markdown("##### Energy Equivalence:")
            st.write(
                f"This estimated daily saving of **{estimated_daily_savings_joules:,.2f} joules** is equivalent to:"
            )

            # X number of bulbs powered for 1 hour
            # Y number of hours for 1 bulb
            # Both X and Y will be the same value here, which is total_bulb_hours_equivalent
            num_bulbs_for_one_hour = total_bulb_hours_equivalent
            num_hours_for_one_bulb = total_bulb_hours_equivalent

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Powering 12W CFL Bulbs (for 1 hour each)",
                    value=f"{num_bulbs_for_one_hour:.1f} bulbs",
                    st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <div style="font-size: 0.875rem; color: rgb(85, 87, 97); line-height: 1.25rem;">{label_bulbs}</div>
                    <div style="font-size: 1.875rem; font-weight: 600; color: green; line-height: 2.25rem;">{value_bulbs}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Energy per bulb-hour is not configured correctly for equivalence calculation.")
    else:
        st.info("No potential savings identified, so no daily impact or energy equivalence to show.")
