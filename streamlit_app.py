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

# --- Azure OpenAI Function (Modified to Extract Percentage) ---
# @st.cache_data # Optionally cache OpenAI responses for a short time
# --- Azure OpenAI Function (Modified to Extract Total Percentage Savings) ---
# @st.cache_data # Optionally cache OpenAI responses for a short time
def get_openai_recommendations(source_code, features_dict):
    """Sends code and features to Azure OpenAI for recommendations and tries to extract total savings percentage."""
    recommendations = "Could not retrieve recommendations." # Default message
    total_potential_savings_percent = 0.0
    extracted_savings = [] # To store individual savings

    try:
        # --- IMPORTANT: Configure Credentials ---
        if not all(k in st.secrets for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]):
            st.error("Azure OpenAI credentials missing in Streamlit Secrets (secrets.toml).")
            st.info("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME are set in .streamlit/secrets.toml")
            return "Credentials configuration missing.", total_potential_savings_percent

        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
        deployment_name = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]

        if not all([api_key, azure_endpoint, api_version, deployment_name]):
            st.error("One or more Azure OpenAI credentials in Streamlit Secrets are empty.")
            return "Credentials configuration incomplete.", total_potential_savings_percent


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
        - Algorithmic efficiency (e.g., unnecessary computations, better data structures)
        - Loop optimizations (e.g., reducing iterations, vectorization)
        - I/O operations (e.g., batching, buffering, efficient file handling)
        - Library usage (e.g., choosing lighter alternatives if possible, efficient use of heavy libraries)
        - Concurrency/Parallelism (potential benefits or overhead)
        - Memory usage patterns

        Provide specific, actionable recommendations on how to modify the code to reduce energy consumption. Focus on practical changes and explain the reasoning. Structure recommendations clearly, perhaps using bullet points.

        Code Features:
        {features_str}

        Source Code:
        ```python
        {source_code}
        Recommendations:

        Also give the **estimated percentage improvement in energy efficiency** for each recommendation, if possible, in the following format at the end of each recommendation: **(Estimated Saving: X-Y%)**. If a single percentage is more appropriate, use that format: **(Estimated Saving: Z%)**. If a percentage cannot be estimated, please omit it.

        At the end. give the value of total enery saved in joules. Example: Total Enery Saved : 21 joules
        """}
        ]

        # --- Make API Call ---
        st.write("_Contacting Azure OpenAI... (This may take a moment)_") # Give user feedback
        response = client.chat.completions.create(
            model=deployment_name, # Your deployment name
            messages=prompt_messages,
            temperature=0.5, # Lower temperature for more focused recommendations
            max_tokens=4000, # Increased slightly for potentially longer recommendations
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        # --- Extract Response and Attempt to Parse Savings ---
        if response.choices:
            recommendations = response.choices[0].message.content.strip()
            if not recommendations: # Handle empty response case
                recommendations = "AI model returned an empty recommendation."
            else:
                # Try to find all percentage savings in the recommendations
                savings_matches = re.findall(r"\(Estimated Saving: (\d+\.?\d*)(?:-(\d+\.?\d*))?%?\)", recommendations)
                for match in savings_matches:
                    lower_bound = float(match[0])
                    upper_bound_str = match[1]
                    if upper_bound_str:
                        upper_bound = float(upper_bound_str)
                        extracted_savings.append((lower_bound + upper_bound) / 2.0) # Take the average of the range
                    else:
                        extracted_savings.append(lower_bound)

                total_potential_savings_percent = sum(extracted_savings)

        else:
            recommendations = "No recommendations received from API (response structure unexpected)."

    except ImportError:
        st.error("The 'openai' library is not installed. Please add it to requirements.txt and reinstall.")
        recommendations = "OpenAI library not found."
    except KeyError as e:
        # This catches cases where a secret isn't defined in secrets.toml
        st.error(f"Azure OpenAI credential '{e}' not found in Streamlit Secrets (secrets.toml).")
        recommendations = f"Credential configuration missing: {e}"
    except Exception as e:
        st.error(f"Error calling Azure OpenAI API: {e}")
        recommendations = f"Error fetching recommendations: {e}"

    return recommendations, total_potential_savings_percent

# --- Process Files (Modified Savings Calculation) ---
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

                            # Get OpenAI Recommendations (Modified to return total savings percent)
                            st.write("💡 **Fetching Energy Saving Recommendations...**")
                            with st.spinner("Contacting Azure OpenAI..."):
                                recommendations, total_savings_percent = get_openai_recommendations(source_code, features_dict)
                            st.markdown("**Recommendations:**")
                            st.markdown(recommendations) # Display recommendations using markdown

                            if total_savings_percent > 0:
                                estimated_saving = prediction * (total_savings_percent / 100.0)
                                st.info(f"**Estimated Potential Saving (Total):** {estimated_saving:.2f} joules (based on AI recommendations totaling up to {total_savings_percent:.0f}% improvement)")
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
    st.metric("Total Scripts Analyzed", st.session_state.get('total_scripts_analyzed', 0))
    st.metric("Total Predicted Energy Consumption", f"{st.session_state.get('total_predicted_consumption', 0):.2f} joules")
    st.metric("Estimated Total Potential Saving (Based on Recommendations)", f"{st.session_state.get('potential_total_savings', 0):.2f} joules")
