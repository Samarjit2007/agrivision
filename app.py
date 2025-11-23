import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from datetime import date
from PIL import Image
import cv2
from sklearn.cluster import KMeans

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Setup ----------------
st.set_page_config(page_title="Aggrivision 2.0", layout="wide")

DATA_DIR = "data"
DATA_FILE = "crop_data.csv"
JOURNAL_FILE = "journal.csv"
CROP_FILE = "Crop_Recommendation.csv"
FERT_FILE = "Fertilizer Prediction.csv"

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# ---------------- CSV Helpers ----------------
def load_csv_dynamic(filename):
    ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def build_baselines():
    path = os.path.join(DATA_DIR, CROP_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    numeric_cols = [c for c in df.columns if c.lower() != "crop" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()

    agg_dict = {c:["min","mean","max"] for c in numeric_cols}
    baselines = df.groupby("Crop").agg(agg_dict).reset_index()
    baselines.columns = ["Crop"] + [f"{col}_{stat}" for col,stat in baselines.columns[1:]]
    return baselines

def load_fert_dataset():
    path = os.path.join(DATA_DIR, FERT_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    rename_map = {
        "Temparature": "Temperature",
        "Moisture": "Soil Moisture",
        "Phosphorous": "Phosphorus"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "Fertilizer Name" not in df.columns:
        st.error("Dataset missing Fertilizer Name column.")
        return pd.DataFrame()

    return df.dropna(subset=["Fertilizer Name"]).drop_duplicates()
    # ---------------- Custom Utility Functions ----------------
def crop_selector(label, df_user_records, df_baseline_data):
    """
    A custom widget for selecting a crop from existing records, baseline data,
    or allowing the user to input a new crop.
    """
    # 1. Get all unique crops from user's journal
    existing_crops = df_user_records["Crop"].unique().tolist() if "Crop" in df_user_records.columns and not df_user_records.empty else []
    
    # Check if the baseline file contains "Crop Name" or just "Crop"
    baseline_col = "Crop Name" if "Crop Name" in df_baseline_data.columns else "Crop"

    # 2. Get crops from the baseline file (Crop_Recommendation.csv)
    baseline_crops = df_baseline_data[baseline_col].unique().tolist() if baseline_col in df_baseline_data.columns and not df_baseline_data.empty else []

    # 3. Combine, sort, and add the "Add New" option
    all_crops = sorted(list(set(existing_crops + baseline_crops)))
    options = ["<Add New Crop>"] + all_crops
    
    # 4. Use Streamlit selectbox
    selected_option = st.selectbox(label, options=options)

    selected_crop = selected_option
    
    # 5. Handle the Add New case with a text input
    if selected_option == "<Add New Crop>":
        new_crop = st.text_input("Enter the new crop name:")
        if new_crop:
            selected_crop = new_crop
        elif all_crops:
            # Fallback to the first crop if user selected <Add New> but entered nothing
            selected_crop = all_crops[0]
        else:
            selected_crop = "" # Return empty string if no crops exist

    return selected_crop

# ---------------- Fertilizer ML ----------------
@st.cache_resource 
def build_fertilizer_model():
    # 1. Load Data
    df = load_fert_dataset() # Assumes load_fert_dataset() is defined and works
    
    if df.empty:
        # Now returns 4 values, including empty list for classes
        return None, 0.0, [], [] 

    # 2. DATA PREPROCESSING AND SELECTION
    features = ['Temperature', 'Humidity', 'Soil Moisture', 'Soil Type', 
                'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorus']
    
    df_cleaned = df[features + ['Fertilizer Name']].dropna()
    
    X = df_cleaned[features]
    y = df_cleaned['Fertilizer Name']
    
    # 3. SEPARATE NUMERIC AND CATEGORICAL COLUMNS
    categorical_features = ['Soil Type', 'Crop Type']

    # 4. ONE-HOT ENCODING (converts text columns to numbers)
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    
    # Store the final feature names for use during prediction (CRUCIAL FIX)
    trained_features = X_encoded.columns.tolist()

    # 5. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # 6. CREATE PIPELINE
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 7. TRAIN MODEL
    pipeline.fit(X_train, y_train)
    
    # 8. EVALUATE
    accuracy = pipeline.score(X_test, y_test)
    
    # Get the list of all unique class labels (fertilizer names)
    fertilizer_classes = pipeline['clf'].classes_.tolist()
    
    # Return 4 values: pipeline, accuracy, feature names, AND classes
    return pipeline, accuracy, trained_features, fertilizer_classes


# ---------------- Prediction Logic (FIXES THE COLUMN ALIGNMENT ERROR AND PROVIDES TOP 3) ----------------
def predict_fertilizer(model, input_data, trained_features, classes, top_n=3):
    """
    Predicts fertilizer using probability and returns the top N suggestions.
    """
    # 1. Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. ONE-HOT ENCODE the input data for 'Soil Type' and 'Crop Type'
    categorical_features = ['Soil Type', 'Crop Type']
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)

    # 3. ALIGN COLUMNS: The critical fix for feature mismatch
    missing_cols = list(set(trained_features) - set(input_encoded.columns))
    
    for col in missing_cols:
        input_encoded[col] = 0
        
    # 4. Final step: Ensure the order of columns matches the training order exactly
    input_final = input_encoded[trained_features] 
    
    # Use predict_proba to get probabilities for all classes (the fix for top 3!)
    probabilities = model.predict_proba(input_final)[0]
    
    # Combine classes and probabilities into a series
    result_series = pd.Series(probabilities, index=classes)
    
    # Sort and get the top N results
    top_suggestions = result_series.nlargest(top_n)
    
    # Format the results into a list of tuples: [(name, probability), ...]
    top_list = [(name, prob) for name, prob in top_suggestions.items()]
    
    return top_list
# ---------------- Navigation ----------------
page = st.sidebar.radio("Go to", [
    "Dashboard","Data entry","Assess","Journal","Settings","Fertilizer Advisor"
])

# ---------------- Dashboard ----------------
# NOTE: This block must start with 'if' if it is the first page check, 
# otherwise use 'elif' if it follows another 'if' or 'elif'.
if page == "Dashboard": 
    st.title("📊 Sensor Data Dashboard")
    
    # Loads the full dataset using the robust function you have defined globally
    # df_plot requires the load_csv_dynamic and DATA_FILE variables defined elsewhere
    df_plot = load_csv_dynamic(DATA_FILE) 
    
    if len(df_plot) == 0:
        st.warning("No data found to plot. Please use the Data Entry page first.")
    else:
        # Get the unique crops for filtering
        available_crops = df_plot['Crop'].unique().tolist()
        
        # Crop Selector
        selected_crop = st.selectbox(
            "Filter Data by Crop:",
            options=available_crops
        )
        
        # Filter the data for the selected crop
        df_filtered = df_plot[df_plot['Crop'] == selected_crop]

        # Dynamically determine metric columns based on the data, excluding Date and Crop
        # This fixes the NameError: name 'dfc' is not defined from before
        metrics_cols = [c for c in df_plot.columns if c not in ["Date", "Crop"]]
        
        # NEW: Add the comparative option to the list
        plot_options = ["All Parameters (Comparative)"] + metrics_cols 

        # Parameter Selector
        selected_metric = st.selectbox(
            "Select Parameter to Visualize:",
            options=plot_options
        )

        st.divider()
# --- NEW METRICS WIDGETS ---
        if not df_filtered.empty:
            # Get the latest row for up-to-date readings
            latest_row = df_filtered.sort_values(by='Date', ascending=False).iloc[0]
            
            # Calculate Averages for environmental factors
            avg_temp = df_filtered['Temperature'].mean()
            avg_ph = df_filtered['pH_Value'].mean()
            
            st.subheader(f"Current Status for {selected_crop} (Latest Reading: {latest_row['Date']})")

            # Layout metrics in columns
            colN, colP, colK, colTemp, colpH = st.columns(5)
            
            # Latest readings for Nutrients
            colN.metric("Latest Nitrogen (N)", f"{latest_row['Nitrogen']:.1f}")
            colP.metric("Latest Phosphorus (P)", f"{latest_row['Phosphorus']:.1f}")
            colK.metric("Latest Potassium (K)", f"{latest_row['Potassium']:.1f}")
            
            # Average metrics for Environmental Factors
            colTemp.metric("Avg Temp (°C)", f"{avg_temp:.1f}")
            colpH.metric("Avg pH", f"{avg_ph:.2f}")

        st.divider() 
        # --- END METRICS WIDGETS ---

        # NOTE: The rest of the code (metrics_cols definition, st.selectbox for plotting, and the graphing logic) 
        # should follow immediately after the st.divider().
        # Logic for the NEW Comparative Chart
        if selected_metric == "All Parameters (Comparative)":
            st.subheader(f"Comparative Trends for {selected_crop}")

            # 1. Prepare data for comparative plot (Melting/Unpivoting)
            # This is essential for Altair to plot multiple lines from different columns
            df_melted = df_filtered.melt(
                id_vars=['Date'], 
                value_vars=metrics_cols,
                var_name='Parameter', 
                value_name='Value'
            )
            
            # 2. Create the Altair Comparative Line Chart
            chart = alt.Chart(df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Date'), 
                y=alt.Y('Value:Q', title='Value'),
                # Use 'Parameter' column to assign a different color/line to each metric
                color=alt.Color('Parameter:N', title='Parameter'),
                tooltip=['Date:T', 'Parameter:N', 'Value:Q']
            ).properties(
                title=f"Comparative Trends for {selected_crop}"
            ).interactive() # Allows zooming and panning
            
            st.altair_chart(chart, use_container_width=True)

        # Logic for the ORIGINAL Single Parameter Chart (Preserved)
        else:
            st.subheader(f"Time Series for {selected_metric} ({selected_crop})")
            
            # This uses the standard Streamlit line chart function
            st.line_chart(df_filtered, x='Date', y=selected_metric)
# ---------------- Data entry ----------------
elif page == "Data entry":
    st.title("📝 Enter crop data")
    
    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 Bulk Upload"])
    
    df_existing = load_csv_dynamic(DATA_FILE)
    df_baseline_file = pd.read_csv(os.path.join(DATA_DIR, CROP_FILE)) if os.path.exists(os.path.join(DATA_DIR, CROP_FILE)) else pd.DataFrame()

    # --- TAB 1: Manual Entry (Your existing logic) ---
    with tab1:
        crop = crop_selector("Crop name (choose or add new)", df_existing, df_baseline_file)
        date_val = st.date_input("Date", value=date.today(), help="Select the date of recording.")

        baselines = build_baselines()
        baseline_row = baselines[baselines["Crop"] == crop].iloc[0] if (len(baselines) > 0 and crop in baselines["Crop"].values) else None

        with st.form("manual_entry_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                n = st.number_input("Nitrogen (N)", value=float(baseline_row["Nitrogen_mean"]) if baseline_row is not None and "Nitrogen_mean" in baseline_row else 0.0)
                p = st.number_input("Phosphorus (P)", value=float(baseline_row["Phosphorus_mean"]) if baseline_row is not None and "Phosphorus_mean" in baseline_row else 0.0)
                k = st.number_input("Potassium (K)", value=float(baseline_row["Potassium_mean"]) if baseline_row is not None and "Potassium_mean" in baseline_row else 0.0)
            with col2:
                temp = st.number_input("Temperature (°C)", value=float(baseline_row["Temperature_mean"]) if baseline_row is not None and "Temperature_mean" in baseline_row else 0.0)
                hum = st.number_input("Humidity (%)", value=float(baseline_row["Humidity_mean"]) if baseline_row is not None and "Humidity_mean" in baseline_row else 0.0)
                ph = st.number_input("pH Level", value=float(baseline_row["pH_Value_mean"]) if baseline_row is not None and "pH_Value_mean" in baseline_row else 7.0)
            with col3:
                rain = st.number_input("Rainfall (mm)", value=float(baseline_row["Rainfall_mean"]) if baseline_row is not None and "Rainfall_mean" in baseline_row else 0.0)
            
            submitted = st.form_submit_button("Save Entry")
            if submitted:
                new_row = {
                    "Date": str(date_val), "Crop": crop, 
                    "Nitrogen": n, "Phosphorus": p, "Potassium": k, 
                    "Temperature": temp, "Humidity": hum, "pH_Value": ph, "Rainfall": rain
                }
                df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
                df_existing.to_csv(os.path.join(DATA_DIR, DATA_FILE), index=False)
                st.success(f"Saved entry for {crop} on {date_val}.")

    # --- TAB 2: Bulk Upload (New Feature) ---
    with tab2:
        st.info("Upload a CSV or Excel file with columns like: Nitrogen, Phosphorus, Potassium, Temperature, etc.")
        
        uploaded_file = st.file_uploader("Choose file", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                # Load file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df_new = pd.read_csv(uploaded_file)
                else:
                    df_new = pd.read_excel(uploaded_file)
                
                # Clean column names (strip spaces, handle case)
                df_new.columns = [c.strip() for c in df_new.columns]
                
                # Smart Auto-fill for missing columns
                # 1. If "Crop" is missing in file, let user pick one for the whole batch
                if "Crop" not in df_new.columns:
                    batch_crop = st.selectbox("Crop column missing in file. Select crop for this batch:", 
                                              options=sorted(df_existing["Crop"].unique().tolist()) + ["<Add New>"])
                    if batch_crop == "<Add New>":
                        batch_crop = st.text_input("Enter new crop name for batch")
                    if batch_crop:
                        df_new["Crop"] = batch_crop
                
                # 2. If "Date" is missing, use today
                if "Date" not in df_new.columns:
                    df_new["Date"] = str(date.today())
                
                # Preview
                st.write("Preview of data to add:")
                st.dataframe(df_new.head())
                
                # Validate required columns (soft check)
                required = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]
                missing = [c for c in required if c not in df_new.columns]
                
                if missing:
                    st.warning(f"⚠️ Your file is missing these columns: {missing}. They will be filled with 0.")
                
                if st.button("Confirm & Append Data"):
                    # Fill missing cols with 0
                    for c in required:
                        if c not in df_new.columns:
                            df_new[c] = 0.0
                    
                    # Align columns and save
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(os.path.join(DATA_DIR, DATA_FILE), index=False)
                    st.success(f"Successfully added {len(df_new)} records!")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # --- Display Data (Shared) ---
    st.divider()
    df_final = load_csv_dynamic(DATA_FILE)
    if len(df_final) > 0:
        st.subheader("📄 Recorded Data")
        st.dataframe(df_final.sort_values("Date", ascending=False), use_container_width=True)
        
        # Deletion Logic
        with st.expander("Delete Records"):
            idx = st.selectbox("Select index to delete", df_final.index)
            if st.button("Delete selected"):
                df_final = df_final.drop(idx)
                df_final.to_csv(os.path.join(DATA_DIR, DATA_FILE), index=False)
                st.experimental_rerun()

# ---------------- Assess ----------------
# ---------------- Assess (Advanced) ----------------
elif page == "Assess":
    st.title("🌿 Intelligent Leaf Analysis")
    st.markdown("Upload a leaf photo to detect **disease spots**, **chlorosis** (yellowing), and calculate a **health score**.")

    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Load and Preprocess
        # Convert PIL Image to OpenCV format (RGB -> BGR)
        pil_image = Image.open(uploaded_file).convert("RGB")
        img = np.array(pil_image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 2. Advanced Segmentation (Isolate Leaf)
        # Convert to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Define Color Ranges
        # Green (Healthy)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Yellow (Chlorosis / Nitrogen Def)
        lower_yellow = np.array([15, 40, 40])
        upper_yellow = np.array([35, 255, 255])
        
        # Brown (Necrosis / Disease) - covers two ranges (0-15 and 160-180)
        lower_brown1 = np.array([0, 20, 20])
        upper_brown1 = np.array([15, 255, 255])
        lower_brown2 = np.array([160, 20, 20])
        upper_brown2 = np.array([180, 255, 255])

        # Create Masks
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_brown = cv2.inRange(hsv, lower_brown1, upper_brown1) | cv2.inRange(hsv, lower_brown2, upper_brown2)
        
        # Combine disease masks
        mask_disease = cv2.bitwise_or(mask_yellow, mask_brown)
        
        # Total Leaf Area (Green + Yellow + Brown)
        # Note: This ignores background if background is white/black (neither green/yellow/brown)
        mask_leaf = cv2.bitwise_or(mask_green, mask_disease)
        total_pixels = cv2.countNonZero(mask_leaf)
        
        # 3. Spot Detection (Contours)
        contours, _ = cv2.findContours(mask_disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_count = len(contours)
        
        # Draw contours on image for visualization (Red outline)
        img_overlay = img.copy()
        cv2.drawContours(img_overlay, contours, -1, (255, 0, 0), 2)
        
        # 4. K-Means Dominant Color Extraction
        # Flatten the leaf pixels (exclude background)
        if total_pixels > 0:
            # Extract pixels where mask_leaf is non-zero
            masked_pixels = img[mask_leaf > 0]
            
            # Use KMeans to find top 5 colors
            if len(masked_pixels) > 100: # Need enough pixels
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(masked_pixels)
                dominant_colors = kmeans.cluster_centers_.astype(int)
            else:
                dominant_colors = []
        else:
            dominant_colors = []

        # 5. Metrics & Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_overlay, caption=f"Analyzed Image ({spot_count} spots detected)", use_column_width=True)
            
            # Display Dominant Colors Palette
            if len(dominant_colors) > 0:
                st.caption("Dominant Colors Found:")
                palette_cols = st.columns(5)
                for i, color in enumerate(dominant_colors):
                    # Color swatch
                    with palette_cols[i]:
                        st.markdown(
                            f'<div style="background-color: rgb({color[0]},{color[1]},{color[2]}); height: 30px; border-radius: 5px;"></div>',
                            unsafe_allow_html=True
                        )

        with col2:
            st.subheader("📊 Analysis Report")
            
            if total_pixels == 0:
                st.warning("Could not detect a leaf. Try a clearer photo with a plain background.")
            else:
                # Calculate percentages
                green_area = cv2.countNonZero(mask_green)
                yellow_area = cv2.countNonZero(mask_yellow)
                brown_area = cv2.countNonZero(mask_brown)
                
                pct_green = (green_area / total_pixels) * 100
                pct_yellow = (yellow_area / total_pixels) * 100
                pct_brown = (brown_area / total_pixels) * 100
                
                # Health Score (Simple weighted formula)
                # 100% green = 100 score. Yellow/Brown reduce score.
                health_score = max(0, 100 - (pct_yellow * 1.0) - (pct_brown * 1.5))
                
                st.metric("Health Score", f"{health_score:.1f} / 100", delta=f"{pct_green:.1f}% Healthy Tissue")
                
                st.progress(int(health_score), text="Overall Health")
                
                st.write("---")
                st.write(f"**🍂 Disease Spots:** {spot_count}")
                st.write(f"**🟡 Yellowing (Chlorosis):** {pct_yellow:.1f}%")
                st.write(f"**🟤 Necrosis (Dead Tissue):** {pct_brown:.1f}%")
                
                st.write("---")
                st.subheader("🩺 Recommendation")
                if health_score > 85:
                    st.success("✅ Plant looks healthy! Maintain current care.")
                elif pct_yellow > pct_brown:
                    st.warning("⚠️ High Yellowing detected. This often indicates **Nitrogen deficiency** or **Overwatering**. Check soil moisture.")
                elif pct_brown > 5:
                    st.error("🚨 Necrosis detected. Large brown spots often indicate **Fungal infection** or **Sun scorch**. Isolate plant and consider fungicide.")
                else:
                    st.info("ℹ️ Mild stress detected. Monitor closely.")

# ---------------- Journal ----------------
elif page == "Journal":
    st.title("📝 Farmer’s journal")
    df_user = load_csv_dynamic(DATA_FILE)
    df_baseline = pd.read_csv(os.path.join(DATA_DIR, CROP_FILE)) if os.path.exists(os.path.join(DATA_DIR, CROP_FILE)) else pd.DataFrame()
    crop = crop_selector("Crop", df_user, df_baseline)

    date_val = st.date_input("Date", value=date.today())
    note = st.text_area("Observation / Notes")

    if st.button("Save note"):
        df = load_csv_dynamic(JOURNAL_FILE)
        new_row = {"Date":str(date_val),"Crop":crop,"Note":note.strip()}
        df = pd.concat([df,pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(os.path.join(DATA_DIR, JOURNAL_FILE), index=False)
        st.success("Note saved.")

    df_notes = load_csv_dynamic(JOURNAL_FILE)
    if len(df_notes) > 0:
        st.subheader("📖 Past notes")
        st.dataframe(df_notes.sort_values("Date", ascending=False), use_container_width=True)

        # Delete option for notes
        idx = st.selectbox("Select note index to delete", df_notes.index)
        if st.button("Delete selected note"):
            df_notes = df_notes.drop(idx)
            df_notes.to_csv(os.path.join(DATA_DIR, JOURNAL_FILE), index=False)
            st.success("Note deleted.")

# ---------------- Settings ----------------
elif page == "Settings":
    st.title("⚙️ Settings and baselines")

    # Baselines
    baselines = build_baselines()
    if len(baselines) == 0:
        st.error("No baselines available. Ensure Crop_Recommendation.csv is in data/.")
        st.stop()

    st.subheader("Crop baselines (min / mean / max)")
    st.dataframe(baselines, use_container_width=True)
# ---------------- fertilizer adviser ----------------
elif page == "Fertilizer Advisor":
    st.title("🌱 Fertilizer Advisor")
    st.markdown("Input your soil and environment conditions to get a fertilizer recommendation.")

    # --- 1. CALL THE MODEL BUILDER ---
    # Model pipe, accuracy, trained features, and classes are returned from the function in Part 1
    model_pipe, accuracy, trained_features, fertilizer_classes = build_fertilizer_model()

    # Get unique categorical options for Streamlit selectboxes
    df_fert = load_fert_dataset()
    
    if df_fert.empty or model_pipe is None:
        st.error("🚨 CRITICAL ERROR: Fertilizer data is missing or corrupted. Model cannot run.")
        return # Stop execution if data is missing
    
    soil_types = sorted(df_fert['Soil Type'].unique().tolist())
    crop_types = sorted(df_fert['Crop Type'].unique().tolist())

    # --- 2. SELECTION WIDGETS (Moved outside the form for dynamic filtering) ---
    st.subheader("Select Crop and Soil Type")
    colA, colB = st.columns(2)
    
    with colA:
        # These values are now constantly available for filtering the preview
        soil_type_selected = st.selectbox("Soil Type", options=soil_types, key="preview_soil")
    with colB:
        crop_type_selected = st.selectbox("Crop Type", options=crop_types, key="preview_crop")

    # --- 3. DYNAMIC DATA PREVIEW (The fix for your requirement) ---
    with st.expander(f"📊 View Data for **{crop_type_selected}** on **{soil_type_selected}**"):
        
        # Filter the full DataFrame based on the user's current selections
        df_filtered = df_fert[
            (df_fert['Soil Type'] == soil_type_selected) & 
            (df_fert['Crop Type'] == crop_type_selected)
        ]
        
        if not df_filtered.empty:
            # Display only the relevant columns and the filtered data
            st.write(f"Showing **{len(df_filtered)}** records matching your criteria:")
            df_display = df_filtered[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil Moisture', 'Fertilizer Name']]
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("No specific training records found for this Soil Type and Crop Type combination.")

        st.info(f"Model trained with overall accuracy: **{accuracy:.2f}**.")

    st.markdown("---")


    # --- 4. NUMERIC INPUT FORM (The NPK/Weather inputs remain in the form) ---
    with st.form("fert_advisor_form"):
        st.subheader("Enter Current Conditions")
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Nitrogen (N)", min_value=0.0, value=30.0)
            p = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0)
            k = st.number_input("Potassium (K)", min_value=0.0, value=40.0)
        
        with col2:
            temp = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
            moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0)
        
        submitted = st.form_submit_button("Get Recommendation")

        if submitted:
            # 5. GATHER USER INPUTS INTO A DICT (Using the values from the top selectboxes)
            user_input = {
                "Temperature": temp, "Humidity": hum, "Soil Moisture": moisture,
                "Soil Type": soil_type_selected, # Use the value from the top selectbox
                "Crop Type": crop_type_selected, # Use the value from the top selectbox
                "Nitrogen": n, "Potassium": k, "Phosphorus": p
            }

            # 6. CALL THE PREDICTION FUNCTION
            top_suggestions = predict_fertilizer(model_pipe, user_input, trained_features, fertilizer_classes, top_n=3)
            
            st.success(f"✅ Top Recommendation: **{top_suggestions[0][0]}**")
            
            st.subheader("Top 3 Fertilizer Suggestions")
            st.markdown("---")
            
            for i, (name, prob) in enumerate(top_suggestions):
                if i == 0:
                    st.write(f"🥇 **{name}** (Confidence: {prob:.2%})")
                elif i == 1:
                    st.write(f"🥈 {name} (Confidence: {prob:.2%})")
                elif i == 2:
                    st.write(f"🥉 {name} (Confidence: {prob:.2%})")

