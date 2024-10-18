import dash
import dash_table
import webbrowser
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np

# Load CSV file (in-memory, you can adjust it to read from a file system)
df = pd.read_csv('Final Data Rapid_Assessment.csv',delimiter=";")
print(df)
# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Title
    html.H1("Afri Kids Data Visualizer and Analysis Tool",style={'textAlign': 'center'}),
    # File uploader (if you want to make it dynamic)
    # dcc.Upload(id='upload-data', children=html.Button('Upload CSV')),

    # Dropdown for selecting the type of plot
    html.Label('Choose Plot Type:'),
    dcc.Dropdown(
        id='plot-type',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'},
            {'label': 'Bar Plot', 'value': 'bar'},
        ],
        value='scatter'  # Default value
    ),

    # Multi-select dropdown for choosing the features
    html.Label('Select Features:'),
    dcc.Dropdown(
        id='feature-selector',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=False,
        value=df.columns[0]  # Default to first two columns
    ),dcc.Dropdown(
        id='feature-selector2',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=False,
        value=df.columns[0]  # Default to first two columns
    ),

    # Dropdown for analysis (PCA or Linear Regression)
    html.Label('Choose Analysis:'),

    # Plot area
    dcc.Graph(id='graph-output'),
    html.H2("Selection of Features for Model",style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='multi-feature-selector',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=True,
        value=[k for k in df.columns if "Score" in k] # Default to first two columns
    ),html.H2("Selection of Y for Linear Regression",style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='y-selector',
        options=[{'label': col, 'value': col} for col in df.columns],
        multi=False,
        value=df.columns[0]  # Default to first two columns
    ),dcc.Graph(id='graph-output2'),dcc.Dropdown(
        id='analysis-type',
        options=[
            {'label': 'Linear Regression', 'value': 'linear'},
            {'label': 'PCA', 'value': 'pca'},{'label': 'Lasso', 'value': 'lasso'},{'label': 'K-Means', 'value': 'kmeans'},
        ],
        value='linear'  # Default value
    ),
    # Analysis output area,
# Button to trigger the analysis
    html.Button('Run Analysis', id='run-analysis', n_clicks=0,style={'textAlign': 'center'}),
    html.Div(id='analysis-output'),dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        sort_action='native',  # Enables sorting on column headers
        sort_mode='multi',  # Allows sorting by multiple columns
        page_size=10,  # Display 10 rows per page
        filter_action='native',
    ),


])

# Callback for correlation matrix
@app.callback(
    Output('graph-output2', 'figure'),
    [Input('multi-feature-selector', 'value')]
)
def update_graph_correl(multi_feature):
    if not multi_feature or len(multi_feature) < 2:
        return go.Figure()  # Return an empty figure if no features are selected

        # Select the chosen features
    selected_features = df[multi_feature]

    # Separate numeric and categorical columns
    numeric_features = selected_features.select_dtypes(include=[np.number])
    categorical_features = selected_features.select_dtypes(exclude=[np.number])

    # Perform one-hot encoding on the categorical columns
    if not categorical_features.empty:
        one_hot_encoded = pd.get_dummies(categorical_features)
        # Concatenate the numeric and one-hot encoded features
        processed_features = pd.concat([numeric_features, one_hot_encoded], axis=1)
    else:
        processed_features = numeric_features

    if processed_features.empty:
        return go.Figure()  # Return an empty figure if no valid features are available

    # Calculate the correlation matrix
    corr_matrix = processed_features.corr()

    # Generate hover text with index and column names
    hover_text = [
        [f'Index: {corr_matrix.index[i]}<br>Column: {corr_matrix.columns[j]}<br>Value: {corr_matrix.iloc[i, j]:.2f}'
         for j in range(len(corr_matrix.columns))]
        for i in range(len(corr_matrix.index))]

    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        hoverinfo='text',
        text=hover_text,
        colorscale='Viridis'
    ))

    # Update layout
    fig.update_layout(
        title="Correlation Matrix (with One-Hot Encoding)",
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis_nticks=len(processed_features.columns)
    )

    return fig
@app.callback(
    Output('graph-output', 'figure'),
    [Input('plot-type', 'value'),
     Input('feature-selector', 'value'),Input('feature-selector2', 'value')]
)
def update_graph(plot_type, selected_feature,selected_feature2):
    x, y = selected_feature, selected_feature2

    if plot_type == 'scatter':
        fig = px.scatter(df, x=x, y=y)
    elif plot_type == 'line':
        fig = px.line(df, x=x, y=y)
    elif plot_type == 'bar':
        fig = px.bar(df, x=x, y=y)
    return fig

# Callback for running the analysis (PCA or Linear Regression)
@app.callback(
    Output('analysis-output', 'children'),
    [Input('run-analysis', 'n_clicks')],
    [State('analysis-type', 'value'),
     State('multi-feature-selector', 'value'),State('y-selector', 'value')]
)
def run_analysis(n_clicks, analysis_type, selected_feature,y_selector):
    # Drop rows with missing values in the selected columns
    # Select the chosen features and y variable
    # Ensure X and y have the same length after dropping missing values
    # Separate numeric and categorical columns
    # Select the chosen features and y variable
    X = df[selected_feature]
    y = df[y_selector]

    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number])
    categorical_features = X.select_dtypes(exclude=[np.number])

    # Perform one-hot encoding on the categorical columns if present
    if not categorical_features.empty:
        one_hot_encoded = pd.get_dummies(categorical_features)
        # Concatenate the numeric and one-hot encoded features
        X_processed = pd.concat([numeric_features, one_hot_encoded], axis=1)
    else:
        X_processed = numeric_features

    # Fill missing values with the mean before normalization
    X_processed = X_processed.fillna(X_processed.mean())
    y = y.fillna(y.mean())

    # Normalize the data (Z-score normalization)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_processed)

    # Align X and y to ensure they have the same length
    X_aligned, y_aligned = pd.DataFrame(X_normalized).align(y, join='inner', axis=0)


    if analysis_type == 'pca':
        # Perform PCA
        pca = PCA(
            n_components=min(len(X_normalized[0]), len(X_normalized)))  # Ensure component count doesn't exceed columns
        components = pca.fit_transform(X_normalized)

        # Explained variance ratio for each component
        explained_variance = pca.explained_variance_ratio_

        # Factor loadings (eigenvectors scaled by the square root of the eigenvalues)
        factor_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Create a table of explained variance for each component
        explained_variance_df = pd.DataFrame({
            "Principal Component": [f"PC{i + 1}" for i in range(len(explained_variance))],
            "Explained Variance Ratio": explained_variance
        })

        # Create a DataFrame for factor loadings
        factor_loadings_df = pd.DataFrame(
            factor_loadings,
            index=X_processed.columns.tolist(),
            columns=[f"PC{i + 1}" for i in range(factor_loadings.shape[1])]
        )

        # Display the explained variance table using Dash DataTable
        table_variance = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in explained_variance_df.columns],
            data=explained_variance_df.to_dict('records'),
            style_table={'width': '50%', 'margin': 'auto'},
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'}
        )

        # Display the factor loadings table using Dash DataTable
        table_loadings = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in factor_loadings_df.reset_index().columns],
            data=factor_loadings_df.reset_index().to_dict('records'),
            style_table={'width': '80%', 'margin': 'auto'},
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'}
        )

        return html.Div([table_variance, table_loadings])
    elif analysis_type == 'linear':

        # Add a constant term for the intercept in the model
        X_with_const = sm.add_constant(X_normalized)
        # Perform Linear Regression using statsmodels
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        scales = scaler.scale_  # This gives the standard deviations of the original features

        # Unnormalize the coefficients (excluding the intercept)
        unnormalized_coefficients = results.params[1:] / scales

        # The intercept doesn't change, so keep it the same
        unnormalized_intercept = results.params[0]

        # Create a new DataFrame for the unnormalized coefficients
        coef_df = pd.DataFrame({
    'columns names': ['const'] + X_processed.columns.tolist(),
    'coefficients': [round(unnormalized_intercept, 2)] + [round(c, 2) for c in unnormalized_coefficients],
    'standard_errors': [round(se, 2) for se in results.bse],
    't_values': [round(t, 2) for t in results.tvalues],
    'p_values': [round(p, 2) for p in results.pvalues],
    'conf_lower': [round(cl, 2) for cl in results.conf_int()[0]],
    'conf_upper': [round(cu, 2) for cu in results.conf_int()[1]]
    }).sort_values("p_values",ascending=True)

        # Display the table of regression stats using Dash DataTable
        return html.Div([html.H3(f"R2 is : {results.rsquared} and f pvalue of the model : {results.f_pvalue}",style={'textAlign': 'center'}),dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in coef_df.columns],
            data=coef_df.to_dict('records'),
            style_table={'width': '50%', 'margin': 'auto'},
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign': 'flex'}
        )])
    elif analysis_type == 'lasso':

        # Import LassoCV from sklearn
        from sklearn.linear_model import LassoCV

        # Add a constant term for the intercept in the model
        X_with_const = sm.add_constant(X_normalized)

        # Perform LassoCV
        lasso_cv = LassoCV(cv=5, random_state=0).fit(X_normalized, y_aligned)

        # Get coefficients, including the intercept
        coefficients = np.append(lasso_cv.intercept_, lasso_cv.coef_)

        # Assuming scaler is StandardScaler or similar
        scales = scaler.scale_  # This gives the standard deviations of the original features

        # Unnormalize the Lasso coefficients (excluding the intercept)
        unnormalized_coefficients = lasso_cv.coef_ / scales

        # The intercept doesn't change, so keep it the same
        unnormalized_intercept = lasso_cv.intercept_

        # Combine the unnormalized intercept and coefficients
        coefficients_unnormalized = np.append(unnormalized_intercept, unnormalized_coefficients)

        # Create a DataFrame for the unnormalized coefficients
        coef_df = pd.DataFrame({
            'columns names': ['Intercept'] + X_processed.columns.tolist(),
            'coefficients': coefficients_unnormalized
        })

        # Sort the DataFrame by the absolute value of the coefficients in descending order
        coef_df['abs_coefficients'] = coef_df['coefficients'].abs()
        coef_df = coef_df.sort_values(by='abs_coefficients', ascending=False).drop(columns='abs_coefficients')

        # Display the unnormalized and sorted table of LassoCV results
        return html.Div([
            html.H3(
                f"Optimal alpha: {round(lasso_cv.alpha_, 2)} | R²: {round(lasso_cv.score(X_normalized, y_aligned), 2)}",
                style={'textAlign': 'center'}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in coef_df.columns],
                data=coef_df.to_dict('records'),
                style_table={'width': '50%', 'margin': 'auto'},
                style_header={'fontWeight': 'bold'},
                style_cell={'textAlign': 'center'}
            )
        ])
    if analysis_type == 'kmeans':
        # Set number of clusters (you can adjust this to be dynamic if you like)
        n_clusters = 3  # Set this to the number of clusters you want
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_normalized)

        # Create a DataFrame with the original data and cluster labels
        cluster_df = pd.DataFrame(X_processed, columns=X_processed.columns)
        cluster_df['Cluster'] = kmeans.labels_

        # Create a scatter plot to visualize the clusters (using first two selected features)
        fig = px.scatter(cluster_df, x=cluster_df.columns[0], y=cluster_df.columns[1], color='Cluster')

        return html.Div([
            html.H3(f"K-Means Clustering with {n_clusters} Clusters", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in cluster_df.columns],
                data=cluster_df.to_dict('records'),
                style_table={'width': '80%', 'margin': 'auto'},
                style_header={'fontWeight': 'bold'},
                style_cell={'textAlign': 'center'}
            )
        ])
# Run the app
if __name__ == '__main__':
    port = 8052  # Change to a different port
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open_new(url)
    app.run_server(debug=False, port=port)