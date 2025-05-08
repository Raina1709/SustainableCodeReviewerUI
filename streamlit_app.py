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
