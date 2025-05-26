import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="DNA Methylation Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧬 DNA Methylation Prediction App</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["Data Upload & Exploration", "Model Training", "Prediction", "About"])

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def encode_sequence(sequence):
    """Encode DNA sequence into numerical features"""
    # Count nucleotides
    a_count = sequence.count('A')
    t_count = sequence.count('T')
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    
    total_length = len(sequence)
    
    # Calculate ratios
    gc_content = (g_count + c_count) / total_length if total_length > 0 else 0
    at_content = (a_count + t_count) / total_length if total_length > 0 else 0
    
    return {
        'length': total_length,
        'A_count': a_count,
        'T_count': t_count,
        'G_count': g_count,
        'C_count': c_count,
        'GC_content': gc_content,
        'AT_content': at_content,
        'A_ratio': a_count / total_length if total_length > 0 else 0,
        'T_ratio': t_count / total_length if total_length > 0 else 0,
        'G_ratio': g_count / total_length if total_length > 0 else 0,
        'C_ratio': c_count / total_length if total_length > 0 else 0
    }

def create_features_from_sequence(df):
    """Create features from DNA sequence"""
    features_list = []
    
    for sequence in df['sequence']:
        features = encode_sequence(sequence)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    return pd.concat([df.reset_index(drop=True), features_df], axis=1)

# Page 1: Data Upload & Exploration
if page == "Data Upload & Exploration":
    st.markdown('<div class="section-header">📊 Data Upload & Exploration</div>', unsafe_allow_html=True)
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Training Data")
        train_file = st.file_uploader("Choose train.csv file", type="csv", key="train")
        
    with col2:
        st.subheader("Upload Test Data")
        test_file = st.file_uploader("Choose test.csv file", type="csv", key="test")
    
    # Sample data option
    if st.button("🎲 Use Sample Data"):
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample DNA sequences
        bases = ['A', 'T', 'G', 'C']
        sequences = []
        methylation_scores = []
        
        for i in range(n_samples):
            # Generate random sequence
            seq_length = np.random.randint(50, 201)
            sequence = ''.join(np.random.choice(bases, seq_length))
            sequences.append(sequence)
            
            # Generate methylation score based on GC content with some noise
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
            methylation = gc_content * 0.8 + np.random.normal(0, 0.1)
            methylation = np.clip(methylation, 0, 1)
            methylation_scores.append(methylation)
        
        # Create DataFrames
        train_data = pd.DataFrame({
            'sequence': sequences[:800],
            'methylation_score': methylation_scores[:800]
        })
        
        test_data = pd.DataFrame({
            'sequence': sequences[800:],
            'id': range(len(sequences[800:]))
        })
        
        st.session_state.train_data = train_data
        st.session_state.test_data = test_data
        st.session_state.data_loaded = True
        st.success("✅ Sample data loaded successfully!")
    
    # Load uploaded files
    if train_file is not None and test_file is not None:
        try:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    # Display data if loaded
    if st.session_state.data_loaded:
        train_data = st.session_state.train_data
        test_data = st.session_state.test_data
        
        st.markdown('<div class="section-header">📈 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{len(train_data)}</h3><p>Training Samples</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{len(test_data)}</h3><p>Test Samples</p></div>', unsafe_allow_html=True)
        with col3:
            avg_length = train_data['sequence'].str.len().mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_length:.0f}</h3><p>Avg Sequence Length</p></div>', unsafe_allow_html=True)
        with col4:
            if 'methylation_score' in train_data.columns:
                avg_methylation = train_data['methylation_score'].mean()
                st.markdown(f'<div class="metric-card"><h3>{avg_methylation:.3f}</h3><p>Avg Methylation</p></div>', unsafe_allow_html=True)
        
        # Data preview
        st.subheader("📋 Data Preview")
        tab1, tab2 = st.tabs(["Training Data", "Test Data"])
        
        with tab1:
            st.dataframe(train_data.head(10))
            
        with tab2:
            st.dataframe(test_data.head(10))
        
        # Data analysis
        if 'methylation_score' in train_data.columns:
            st.markdown('<div class="section-header">🔍 Data Analysis</div>', unsafe_allow_html=True)
            
            # Create features for analysis
            train_features = create_features_from_sequence(train_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Methylation Score Distribution")
                fig = px.histogram(train_features, x='methylation_score', nbins=30,
                                 title="Distribution of Methylation Scores")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("GC Content vs Methylation")
                fig = px.scatter(train_features, x='GC_content', y='methylation_score',
                               title="GC Content vs Methylation Score")
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            numeric_cols = train_features.select_dtypes(include=[np.number]).columns
            corr_matrix = train_features[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

# Page 2: Model Training
elif page == "Model Training":
    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Exploration' page.")
    else:
        train_data = st.session_state.train_data
        
        if 'methylation_score' not in train_data.columns:
            st.error("❌ Training data must contain 'methylation_score' column.")
        else:
            # Model selection
            st.subheader("🎯 Model Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Select Model:", 
                                        ["Random Forest", "Linear Regression"])
                
            with col2:
                test_size = st.slider("Test Split Ratio:", 0.1, 0.4, 0.2, 0.05)
            
            if model_type == "Random Forest":
                st.subheader("Random Forest Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Number of Estimators:", 50, 500, 100, 50)
                with col2:
                    max_depth = st.slider("Max Depth:", 5, 50, 10, 5)
                with col3:
                    random_state = st.number_input("Random State:", value=42)
            
            # Train model button
            if st.button("🚀 Train Model"):
                with st.spinner("Training model... Please wait."):
                    try:
                        # Create features
                        train_features = create_features_from_sequence(train_data)
                        
                        # Select features for training
                        feature_cols = ['length', 'A_count', 'T_count', 'G_count', 'C_count',
                                      'GC_content', 'AT_content', 'A_ratio', 'T_ratio', 'G_ratio', 'C_ratio']
                        
                        X = train_features[feature_cols]
                        y = train_features['methylation_score']
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Train model
                        if model_type == "Random Forest":
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        else:
                            model = LinearRegression()
                        
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Store model and results
                        st.session_state.model = model
                        st.session_state.feature_cols = feature_cols
                        st.session_state.model_trained = True
                        st.session_state.model_metrics = {
                            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2
                        }
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred
                        
                        st.success("✅ Model trained successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")
            
            # Display results if model is trained
            if st.session_state.model_trained:
                st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
                
                metrics = st.session_state.model_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{metrics["rmse"]:.4f}</h3><p>RMSE</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>{metrics["mae"]:.4f}</h3><p>MAE</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>{metrics["r2"]:.4f}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-card"><h3>{metrics["mse"]:.4f}</h3><p>MSE</p></div>', unsafe_allow_html=True)
                
                # Prediction vs Actual plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(x=st.session_state.y_test, y=st.session_state.y_pred,
                                   title="Predicted vs Actual Values")
                    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                line=dict(dash="dash", color="red"))
                    fig.update_xaxes(title="Actual Values")
                    fig.update_yaxes(title="Predicted Values")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    residuals = st.session_state.y_test - st.session_state.y_pred
                    fig = px.scatter(x=st.session_state.y_pred, y=residuals,
                                   title="Residual Plot")
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_xaxes(title="Predicted Values")
                    fig.update_yaxes(title="Residuals")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for Random Forest)
                if hasattr(st.session_state.model, 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': st.session_state.feature_cols,
                        'importance': st.session_state.model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    fig = px.bar(feature_importance, x='importance', y='feature',
                               orientation='h', title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

# Page 3: Prediction
elif page == "Prediction":
    st.markdown('<div class="section-header">🔮 DNA Methylation Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model Training' page.")
    else:
        # Single sequence prediction
        st.subheader("🧬 Single Sequence Prediction")
        
        sequence_input = st.text_area("Enter DNA Sequence:", 
                                    placeholder="e.g., ATCGATCGATCG...",
                                    height=100)
        
        if st.button("🎯 Predict Methylation"):
            if sequence_input.strip():
                try:
                    # Validate sequence
                    valid_bases = set('ATCG')
                    if not all(base in valid_bases for base in sequence_input.upper()):
                        st.error("❌ Invalid sequence! Only A, T, C, G bases are allowed.")
                    else:
                        # Create features
                        features = encode_sequence(sequence_input.upper())
                        feature_vector = np.array([features[col] for col in st.session_state.feature_cols]).reshape(1, -1)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(feature_vector)[0]
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f'<div class="metric-card"><h2>{prediction:.4f}</h2><p>Predicted Methylation Score</p></div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("Sequence Features:")
                            for feature, value in features.items():
                                st.write(f"**{feature}**: {value:.4f}")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
            else:
                st.warning("⚠️ Please enter a DNA sequence.")
        
        # Batch prediction
        st.markdown('<div class="section-header">📊 Batch Prediction</div>', unsafe_allow_html=True)
        
        if st.session_state.data_loaded and hasattr(st.session_state, 'test_data'):
            test_data = st.session_state.test_data
            
            if st.button("🚀 Predict Test Set"):
                with st.spinner("Making predictions... Please wait."):
                    try:
                        # Create features for test data
                        test_features = create_features_from_sequence(test_data)
                        X_test = test_features[st.session_state.feature_cols]
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(X_test)
                        
                        # Create results dataframe
                        results = test_data.copy()
                        results['predicted_methylation'] = predictions
                        
                        st.session_state.test_predictions = results
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display results
                        st.subheader("📋 Prediction Results")
                        st.dataframe(results.head(20))
                        
                        # Download predictions
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="methylation_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Prediction distribution
                        fig = px.histogram(results, x='predicted_methylation', nbins=30,
                                         title="Distribution of Predicted Methylation Scores")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")

# Page 4: About
elif page == "About":
    st.markdown('<div class="section-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🧬 DNA Methylation Prediction App
    
    This application predicts DNA methylation scores from DNA sequences using machine learning techniques.
    
    ### 📚 What is DNA Methylation?
    DNA methylation is an epigenetic modification that plays a crucial role in gene regulation, 
    development, and disease. It involves the addition of methyl groups to cytosine bases in DNA.
    
    ### 🔬 How It Works
    1. **Feature Extraction**: The app converts DNA sequences into numerical features including:
       - Nucleotide counts (A, T, G, C)
       - GC content ratio
       - AT content ratio
       - Individual nucleotide ratios
       - Sequence length
    
    2. **Machine Learning Models**: 
       - Random Forest Regressor
       - Linear Regression
    
    3. **Prediction**: Uses trained models to predict methylation scores for new sequences
    
    ### 📊 Features
    - **Data Upload & Exploration**: Upload your own datasets or use sample data
    - **Interactive Visualization**: Explore data distributions and correlations
    - **Model Training**: Train and evaluate different machine learning models
    - **Prediction**: Make predictions for single sequences or batch processing
    - **Performance Metrics**: Comprehensive model evaluation with RMSE, MAE, R² score
    
    ### 💡 Usage Tips
    - Ensure DNA sequences contain only A, T, G, C bases
    - Longer sequences generally provide more stable predictions
    - GC content is often correlated with methylation levels
    - Use the feature importance plot to understand which factors drive predictions
    
    ### 🔧 Technical Details
    - Built with Streamlit for interactive web interface
    - Uses scikit-learn for machine learning algorithms
    - Plotly for interactive visualizations
    - Pandas for data manipulation
    
    ### 📈 Model Performance
    The app provides comprehensive metrics to evaluate model performance:
    - **RMSE**: Root Mean Square Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    - **R² Score**: Coefficient of determination (higher is better, max = 1.0)
    - **Residual Analysis**: Helps identify model bias and assumptions
    
    ---
    
    **Note**: This application is for educational and research purposes. 
    For production use, consider additional validation and more sophisticated feature engineering.
    """)
    
    # Additional statistics if data is loaded
    if st.session_state.data_loaded:
        st.markdown('<div class="section-header">📊 Current Session Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Training Samples**: {len(st.session_state.train_data)}")
            
        with col2:
            st.info(f"**Test Samples**: {len(st.session_state.test_data)}")
            
        with col3:
            if st.session_state.model_trained:
                st.success("**Model Status**: Trained ✅")
            else:
                st.warning("**Model Status**: Not Trained ⚠️")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    🧬 DNA Methylation Prediction App | Built with Streamlit
</div>
""", unsafe_allow_html=True)