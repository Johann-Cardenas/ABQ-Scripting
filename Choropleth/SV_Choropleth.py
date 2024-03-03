
import plotly.express as px
import pandas as pd

# Create Dictionary
# Valid for Insterstate Highways
axle_load_limit = {'AL': 36, 'AK': 38, 'AZ': 34, 'AR': 34, 'CA': 34, 'CO': 36, 'CT': 36, 'DE': 36, 'FL': 44,
                   'GA': 40, 'HI': 34, 'ID': 34, 'IL': 34, 'IN': 36, 'IA': 35, 'KS': 34, 'KY': 34, 'LA': 34,
                   'ME': 41, 'MD': 34, 'MA': 36, 'MI': 34, 'MN': 34, 'MS': 34, 'MO': 34, 'MT': 34, 'NE': 34,
                   'NV': 34, 'NH': 36, 'NJ': 34, 'NM': 34.2, 'NY': 36, 'NC': 38, 'ND': 34, 'OH': 34, 'OK': 34,
                   'OR': 34, 'PA': 34, 'RI': 44.8, 'SC': 36.0, 'SD': 34, 'TN': 34, 'TX': 34, 'UT': 34, 'VT': 36,
                   'VA': 34, 'WA': 34, 'WV': 34, 'WI': 34, 'WY': 36}


Scales=['Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds',
        'Blues','Picnic','Rainbow','Portland', 'Jet','Hot','Blackbody',
        'Earth', 'Electric','Viridis','Cividis','Magma','Plasma','Inferno','Turbo']

# create a dataframe with the state codes and Axle Load Limit values
df = pd.DataFrame({'code': list(axle_load_limit.keys()), 
                   'axle_load_limit': list(axle_load_limit.values())})

# Create the Map
fig = px.choropleth(df, locations='code', 
                        locationmode='USA-states', 
                        color='axle_load_limit', 
                        scope='usa',
                        color_continuous_scale=Scales[3])

# Add Code Name
fig.update_traces(text=df['code'])

# Format Plot
fig.update_layout(title={'text': 'Axle Load Limit Across the United States',
                         'font': {'size': 42, 'family': 'Times New Roman'},
                         'x': 0.5,  # center the title
                         'xanchor': 'center'  # center the title
                         })

fig.update_layout(legend={'font': {'size': 28, 'family': 'Times New Roman'} })

fig.update_layout(coloraxis_colorbar={'title': {'text': '',
                                      'font': {'size': 28, 'family': 'Times New Roman'}},
                                      'tickfont': {'size': 34, 'family': 'Times New Roman'} })

fig.update_layout(width=1920, height=1080)

# Plot
fig.show()

# Save Image
fig.write_image("axle_load_limit_map.png", width=1920, height=1080)
