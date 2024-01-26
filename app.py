from jbi100_app.main import app
from jbi100_app.views.menu import make_menu_layout
from jbi100_app.views.scatterplot import Scatterplot

import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc


df = pd.read_csv('jbi100_app/assets/final_credit_0_60.csv', sep=',')
# read data
df1 = pd.read_csv('jbi100_app/assets/NEW_final_credit_0_60_scaled.csv')
df1.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

# for theme dashboard, and duplicate outputs
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks="initial_duplicate")

pio.templates.default = "plotly_dark"

# reverse axis bar chart of income ranges
income_bins = [0, 25000, 50000, 75000, 100000, 150000, 250000, float('inf')]
income_labels = ['0-25k', '25k-50k', '50k-75k', '75k-100k', '100k-150k', '150k-250k', '250k+']
df['income_bin'] = pd.cut(df['Annual_Income'], bins=income_bins, labels=income_labels, right=False)
income_counts = df['income_bin'].value_counts().sort_index()
high_income_df = df[df['income_bin'] == '250k+']
valid_emi_df = high_income_df[high_income_df['Total_EMI_per_month'] != 0]
fig1 = px.bar(income_counts, x=income_counts.values, y=income_counts.index, orientation='h',
              labels={'x': 'Number of Individuals', 'y': 'Income Range'},
              title='<b>Distribution of Annual Incomes</b>', color_discrete_sequence=['#fb9f3a'])
fig1.update_layout(title_text='<b>Distribution of Annual Incomes</b>', title_x=0.5, height=450, width=675)
fig1.update_yaxes(title_text="Income Bin")


# violin plot of 5 occupations
high_income_df = df[df['income_bin'] == '250k+'].copy()
high_income_counts = high_income_df['Occupation'].value_counts().reset_index(name='Count')
high_income_counts.rename(columns={'index': 'Occupation'}, inplace=True)
frequent_occupations = high_income_counts.head(5)['Occupation']
filtered_df = high_income_df[high_income_df['Occupation'].isin(frequent_occupations)]
fig2 = px.violin(filtered_df, x='Occupation', y='Annual_Income', box=True,
                 title='<b>Violin Plot of Annual Income by Occupation</b>',
                 color_discrete_sequence=['#fb9f3a'])
fig2.update_layout(title_text='<b>Violin Plot of Annual Income by Occupation</b>', title_x=0.5, height=450, width=675)
fig2.update_yaxes(title_text="Annual Income")


@app.callback(  # make violin plot interactive
    Output('violin-plot', 'figure', allow_duplicate=True),
    Input('bar-chart', 'clickData')
)
def update_violin_plot(clickData):
    if clickData is None:
        return fig2  # Default figure

    selected_income_range = clickData['points'][0]['label']
    filtered_df = df[df['income_bin'] == selected_income_range]
    occupation_counts = filtered_df['Occupation'].value_counts()  # count occupations
    top_occupations = occupation_counts.head(5).index.tolist()  # top 5
    filtered_df_top_occupations = filtered_df[filtered_df['Occupation'].isin(top_occupations)]

    fig2_ = px.violin(
        filtered_df_top_occupations, x='Occupation', y='Annual_Income', box=True,
        color_discrete_sequence=['#fb9f3a'])

    # make adjustments to how the graph looks
    fig2_.update_layout(title_text='<b>Violin Plot of Annual Income for Top 5 Occupations</b>', title_x=0.5,
        xaxis={
            'title': {'text': 'Occupation', 'font': {'size': 14, 'color': 'white'}},
            'tickfont': {'size': 14, 'color': 'white'}
        },
        yaxis={
            'title': {'text': 'Annual Income', 'font': {'size': 14, 'color': 'white'}},
            'tickfont': {'size': 14, 'color': 'white'}
        },
        height=450, width=675
    )
    fig2_.update_yaxes(title_text="Annual Income")

    return fig2_

@app.callback(  # make the link bidirectional
    Output('bar-chart', 'figure', allow_duplicate=True),
    Input('violin-plot', 'clickData')
)
def update_bar_chart(clickData):
    if clickData is None:
        return fig1
    selected_occupation = clickData['points'][0]['x']
    filtered_df = df[df['Occupation'] == selected_occupation]
    income_counts = filtered_df['income_bin'].value_counts().sort_index()

    fig1_ = px.bar(income_counts, x=income_counts.values, y=income_counts.index, orientation='h',
                   labels={'x': 'Number of Individuals', 'y': 'Income Range'},
                   title='<b>Distribution of Annual Incomes</b>', color_discrete_sequence=['#fb9f3a'])

    fig1_.update_layout(title_text='<b>Distribution of Annual Incomes</b>', title_x=0.5,
        xaxis={
            'title': {'text': 'Number of Individuals', 'font': {'size': 16, 'color': 'white'}},
            'tickfont': {'size': 14, 'color': 'white'}
        },
        yaxis={
            'title': {'text': 'Income Range', 'font': {'size': 14, 'color': 'white'}},
            'tickfont': {'size': 14, 'color': 'white'}
        },
        height=450, width=675
    )
    fig1_.update_yaxes(title_text="Income Bin")

    return fig1_


# scatter plot of monthly investments
income_df = df.copy()
income_df['Percent_Month_Invest'] = (income_df['Amount_invested_monthly'] / income_df[
    'Monthly_Inhand_Salary']) * 100
fig3 = px.scatter(income_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
                  title='<b>Monthly Income vs. Percentage of Monthly Income Invested</b>',
                  labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                          'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
                  color_discrete_sequence=['#fb9f3a'])
fig3.update_layout(title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested</b>', title_x=0.5, height=450, width=675)

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    [Input('bar-chart', 'clickData'),
     Input('violin-plot', 'clickData')]
)
def update_scatter_plot(bar_click, violin_click):
    ctx = dash.callback_context

    if not ctx.triggered_id:
        # no input triggered the callback, return default figure
        fig3.update_traces(opacity=0.5)
        return fig3

    clicked_input = ctx.triggered_id.split('.')[0]

    # if click comes from bar chart
    if clicked_input == 'bar-chart' and 'label' in bar_click['points'][0]:
        selected_income_range = bar_click['points'][0]['label']
        # use a condition that captures the selected_income_range
        income_df = df[df['income_bin'] == selected_income_range].copy()
        income_df['Percent_Month_Invest'] = (income_df['Amount_invested_monthly'] / income_df[
            'Monthly_Inhand_Salary']) * 100
        fig3_ = px.scatter(
            income_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
            title='<b>Monthly Income vs. Percentage of Monthly Income Invested for</b>',
            labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                    'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
            color_discrete_sequence=['#fb9f3a'])
        fig3_.update_traces(opacity=0.5)
        fig3_.update_layout(title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested</b>',
                            title_x=0.5,height=450, width=675)
        return fig3_

    # if click comes from violin plot
    elif clicked_input == 'violin-plot' and 'x' in violin_click['points'][0]:
        selected_profession = violin_click['points'][0]['x']
        profession_df = df[df['Occupation'] == selected_profession].copy()
        profession_df['Percent_Month_Invest'] = (profession_df['Amount_invested_monthly'] / profession_df[
            'Monthly_Inhand_Salary']) * 100
        fig3_ = px.scatter(
            profession_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
            title=f'<b>Monthly Income vs. Percentage of Monthly Income Invested for {selected_profession}</b>',
            labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                    'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
            color_discrete_sequence=['#fb9f3a'])
        fig3_.update_traces(opacity=0.5)
        fig3_.update_layout(
            title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested for {selected_income_range}</b>',
            title_x=0.5, height=450, width=675)
        return fig3_

    # for some reason no click, handle unexpected situation
    else:
        fig3.update_traces(opacity=0.5)
        return fig3


def create_radar_chart(df1):
    age_labels = ['0-19', '20-29', '30-39', '40-49', '50+']
    radar_cols = ['SC_Monthly_Inhand_Salary', 'SC_Total_EMI_per_month', 'SC_Avg_Monthly_Salary',
                  'SC_Amount_invested_monthly']
    data_radar = df1[radar_cols].copy()
    list_radar = data_radar.values.tolist()
    hover_cols = ['Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Avg_Monthly_Salary', 'Amount_invested_monthly']
    data_hover = df1[hover_cols].copy()
    list_hover = data_hover.values.tolist()
    categories = ['SC_Monthly_Inhand_Salary', 'SC_Total_EMI_per_month', 'SC_Avg_Monthly_Salary',
                  'SC_Amount_invested_monthly']
    labels = ['Monthly Inhand Salary', 'Total EMI per Month', 'Adjusted Monthly Salary', 'Amount Invested Monthly']
    age_labels = df1['age_category'].unique()

    # get colors, reverse them so dark colors are at the back
    color_sequence = px.colors.sequential.Plasma_r[::-1]

    # to make sure sequential colors are not too similiar
    skip_interval = 2

    fig5 = go.Figure()
    for i in range(len(list_radar)):
        # calculate the index by skipping the interval
        color_index = (i * skip_interval) % len(color_sequence)

        # select color from sequence
        color = color_sequence[color_index]

        # when you hover you want the actual value to show
        rounded_actual_values = [round(val, 2) for val in list_hover[i]]
        text_values = [f'{label}: ${value}' for label, value in zip(labels, rounded_actual_values)]
        # add "Scaled Value:" prefix to the hover text for the scaled values
        text_values_scaled = [f'Scaled {label}: {value:.2f}' for label, value in zip(labels, list_radar[i])]
        # combine the text values, br is for a line break in between the values
        combined_text_values = [f"{text_values_scaled[j]}<br>{text_values[j]}" for j in range(len(labels))]

        fig5.add_trace(go.Scatterpolar(
            r=list_radar[i],
            theta=labels,
            fill='toself',
            name=f'{age_labels[i]}',
            text=combined_text_values,
            # Set the color
            marker=dict(color=color)
        ))

    fig5.update_layout(title_text='<b>Radar Chart</b>', title_x=0.46,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5.2]
            )
        ),
        showlegend=True,
        height=450, width=675,
        legend=dict(
            title=dict(text='Age Categories')
        ), )

    return fig5

@app.callback(
    Output('radar-chart', 'figure', allow_duplicate=True),
    [Input('scatter-plot', 'selectedData'),
     Input('reset-button', 'n_clicks')]
)
def update_radar_chart(selecteddata, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered_id and n_clicks > 0:
        return create_radar_chart(df1)

    # avoids issue with button
    if selecteddata is None:
        return create_radar_chart(df1)

    points = selecteddata['points']
    x_vals_selected = [point['x'] for point in points]
    radar_df = df[df['Monthly_Inhand_Salary'].isin(x_vals_selected)]
    radar_df['Avg_Monthly_Salary'] = radar_df.Annual_Income / 12

    # to avoid errors
    if radar_df.empty:
        return create_radar_chart(df1)

    age_bins = [0, 20, 30, 40, 50, float('inf')]
    age_labels = ['0-19', '20-29', '30-39', '40-49', '50+']
    radar_df['age_category'] = pd.cut(radar_df['Age'], bins=age_bins, labels=age_labels, right=False)
    radar_df['age_category'] = radar_df['age_category'].astype(str)
    df_group = radar_df.groupby('age_category').median(numeric_only=True).copy()
    df_group = df_group.reset_index()
    columns_to_scale = ['Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Annual_Income', 'Amount_invested_monthly',
                        'Avg_Monthly_Salary']
    scaler = MinMaxScaler(feature_range=(1, 5))

    # also to avoid errors
    if df_group.empty:
        return create_radar_chart(df1)

    scaled_values = scaler.fit_transform(df_group[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=[f'SC_{col}' for col in columns_to_scale])
    result_df = pd.concat([df_group, scaled_df], axis=1)
    return create_radar_chart(result_df)

# function for reset button
@app.callback(
    Output('bar-chart', 'figure', allow_duplicate=True),
    Output('violin-plot', 'figure', allow_duplicate=True),
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Output('radar-chart', 'figure', allow_duplicate=True),
    Input('reset-button', 'n_clicks')
)
def reset_graphs(n_clicks):
    if n_clicks is None or n_clicks == 0:
        # if the button is not clicked, or it's the first time, return the initial figures
        return fig1, fig2, fig3, create_radar_chart(df1)

    # button is clicked, return the initial figures (aka reset)
    return fig1, fig2, fig3, create_radar_chart(df1)


app.layout = html.Div(style={'backgroundColor': '#111111'}, children=[
    html.Div([
        html.H1(children='The Credit Canvas', style={'textAlign': 'center', 'color': 'white'}),
        html.Div(children='Exploring the credit scene like never before!', style={'textAlign': 'center', 'color': 'white'}),
        dbc.Button('RESET', id='reset-button', n_clicks=0, size="lg",
                   color="light", className="me-1",
                   style={'padding': 10, 'fontWeight': 'bold', 'fontSize': 16, 'margin-left': '250px',
                                 'verticalAlign': 'middle'})
    ]),
    html.Div([
        dcc.Graph(id='bar-chart', figure=fig1),
        dcc.Graph(id='violin-plot', figure=fig2)
    ], style={'margin': 'auto', 'display': 'flex', 'flex': 3, 'justify-content': 'center'}),

    html.Div([
        dcc.Graph(id='scatter-plot', figure=fig3),
        dcc.Graph(id='radar-chart', figure=create_radar_chart(df1))
    ], style={'margin': 'auto', 'display': 'flex', 'justify-content': 'center'})
])


if __name__ == '__main__':
    app.run_server(debug=True, port=8058)