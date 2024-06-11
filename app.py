from dash import Dash, dcc, html, Input, Output, no_update
import pandas as pd
import plotly.express as px


df_shap_pca = pd.read_csv('./assets/sfm_res.csv')

# Create Dash app
app = Dash(__name__)
#app.config.update({
#    'requests_pathname_prefix': '/Interactive_pca/'
#})
# Define colors for each class
colors = {'Antagonist': 'green', 'Agonist': 'orange'}

# Create scatter plot with hover labels
fig = px.scatter(df_shap_pca, x='PC1', y='PC2', color='ligand type', symbol='classification',
                 labels={'y_true': 'True Class'},
                 hover_name='lig_name', hover_data=['y_pred'],
                 color_discrete_map=colors,
                 symbol_sequence=['circle', 'x'],
                 size_max=20)

fig.update_traces(hovertemplate='<b>%{hovertext}</b><br><br>')
# Customize plot layout
fig.update_layout(title='Average Shap values', xaxis_title='PC1', yaxis_title='PC2',
                  coloraxis_showscale=False)

# Define app layout
app.layout = html.Div([
    dcc.Graph(id="scatter-plot", figure=fig),
    html.Div(id="image-container", style={'display': 'flex', 'flexWrap': 'wrap'}),
    html.Div(id="hidden-div", style={'display': 'none'})
])

# Store clicked data and corresponding images
clicked_data = []

# Define callback to display images on dot click
@app.callback(
    Output("image-container", "children"),
    Input("scatter-plot", "clickData")
)
def display_image(clickData):
    global clicked_data
    
    if clickData is None:
        return no_update
    
    hover_text = clickData['points'][0]['hovertext']
    
    # Find the corresponding row in the DataFrame based on the hover text
    data_row = df_shap_pca[df_shap_pca["lig_name"] == hover_text].iloc[0]
    
    # Extract information from the DataFrame
    name = data_row['lig_name']
    
    # Construct image URL
    img_url = f"./assets/{name}/waterfall_plt.png"  # Assuming images are stored in the 'assets' folder
    
    # Create a div to display the image
    image_div = html.Div([
        html.Img(src=img_url, style={"width": "400px","height": "370px"  }),   #{"width": "350px","height": "270px"  }),
        #html.H4(name),
    ], style={'margin': '5px'})  # Add margin to the div
    
    # Append clicked data and corresponding image to the list
    clicked_data.append(image_div)
    
    # Clear the list after the third image
    if len(clicked_data) == 5:
        clicked_data.clear()
    
    # Return all the clicked images
    return clicked_data

if __name__ == '__main__':
   # app.run_server(debug=True, port=8051)
    app.run_server(debug=True, host='0.0.0.0', port=8050)
