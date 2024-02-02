import pandas as pd
import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from dash import html, dcc
import dash_bootstrap_components as dbc

df = pd.read_csv('jbi100_app/assets/final_credit_0_60.csv', sep=',')
df_scale = df.copy()

# for theme dashboard (font), and to allow duplicate outputs in the callback
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks="initial_duplicate")
# makes all figures made by plotly in plotly_dark theme
pio.templates.default = "plotly_dark"


def main():
    """
    Function plots the initial bar chart, violin plot and scatter plot

    :return: Bar chart showing the distribution of annual incomes based on all data
    :return: Violin plot showing the distribution of annual incomes by occupation based on all data
    :return: Scatter plot showing the monthly income vs the percentage of that that is invested based on all data
    """
    # plot the initial bar chart that uses the whole dataset:
    # create the bins and labels
    income_bins = [0, 25000, 50000, 75000, 100000, 150000, 250000, float('inf')]
    income_labels = ['0-25k', '25k-50k', '50k-75k', '75k-100k', '100k-150k', '150k-250k', '250k+']
    # apply the created bins to dataset by making a new column and adding the corresponding label
    df['income_bin'] = pd.cut(df['Annual_Income'], bins=income_bins, labels=income_labels, right=False)
    income_counts = df['income_bin'].value_counts().sort_index()

    # plot the bar chart
    fig_1 = px.bar(income_counts, x=income_counts.values, y=income_counts.index, orientation='h',
                   labels={'x': 'Number of Individuals', 'y': 'Income Range'},
                   title='<b>Distribution of Annual Incomes</b>', color_discrete_sequence=['#fb9f3a'])
    fig_1.update_layout(title_text='<b>Distribution of Annual Incomes</b>', title_x=0.5,
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

    # plot the initial violin plot that uses the whole dataset:
    # violin plot of 5 occupations
    df_prof = df['Occupation'].value_counts().reset_index(name='Count')
    df_prof.rename(columns={'index': 'Occupation'}, inplace=True)
    frequent_occupation = df_prof.head(5)['Occupation']
    df_filtered = df[df['Occupation'].isin(frequent_occupation)]

    fig_2 = px.violin(df_filtered, x='Occupation', y='Annual_Income', box=True,
                      title='<b>Violin Plot of Annual Income by Occupation</b>',
                      color_discrete_sequence=['#fb9f3a'])
    # make adjustments to how the graph looks
    fig_2.update_layout(title_text='<b>Violin Plot of Annual Income for Top 5 Occupations</b>', title_x=0.5,
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
    fig_2.update_yaxes(title_text="Annual Income")

    # plot the scatter plot based on all the data:
    # prepare the data and compute percentage of monthly income invested
    income_df = df.copy()
    income_df['Percent_Month_Invest'] = (income_df['Amount_invested_monthly'] / income_df[
        'Monthly_Inhand_Salary']) * 100
    # plot the scatter plot
    fig_3 = px.scatter(
        income_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
        title=f'<b>Monthly Income vs. Percentage of Monthly Income Invested</b>',
        labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
        color_discrete_sequence=['#fb9f3a'])
    fig_3.update_traces(opacity=0.5)
    fig_3.update_layout(
        title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested}</b>',
        title_x=0.5, height=450, width=675)

    # create a scaled dataframe for the radar chart
    age_bins = [0, 20, 30, 40, 50, float('inf')]
    age_labels = ['0-19', '20-29', '30-39', '40-49', '50+']
    # add the age category to each row
    df_scale['age_category'] = pd.cut(df_scale['Age'], bins=age_bins, labels=age_labels, right=False)
    df_scale['age_category'] = df_scale['age_category'].astype(str)
    # group by the age category
    df_group = df_scale.groupby('age_category').median(numeric_only=True).copy()
    # adjusted monthly salary for irregular jobs
    df_group['Avg_Monthly_Salary'] = df_group.Annual_Income / 12
    df_group = df_group.reset_index()
    # scale the following columns from the data frame
    columns_to_scale = ['Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Annual_Income', 'Amount_invested_monthly',
                        'Avg_Monthly_Salary']
    scaler = MinMaxScaler(feature_range=(1, 5))

    # create dataframe to use as input for function
    scaled_values = scaler.fit_transform(df_group[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=[f'SC_{col}' for col in columns_to_scale])
    result_df = pd.concat([df_group, scaled_df], axis=1)

    return fig_1, fig_2, fig_3, result_df


# call main to make sure that initial graphs are displayed and that figures/df can be accessed inside other functions
fig1, fig2, fig3, scale_df = main()


def create_radar_chart(df_scale_vals):
    """
    Function that creates the initial radar chart containing all data

    :param df_scale_vals: dataframe with scaled entries for radar chart
    :return: radar chart showing multiple attributes per age category
    """
    # prepare data, column names, axis names
    radar_cols = ['SC_Monthly_Inhand_Salary', 'SC_Total_EMI_per_month', 'SC_Avg_Monthly_Salary',
                  'SC_Amount_invested_monthly']
    data_radar = df_scale_vals[radar_cols].copy()
    list_radar = data_radar.values.tolist()
    hover_cols = ['Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Avg_Monthly_Salary', 'Amount_invested_monthly']
    data_hover = df_scale_vals[hover_cols].copy()
    list_hover = data_hover.values.tolist()
    labels = ['Monthly Inhand Salary', 'Total EMI per Month', 'Adjusted Monthly Salary', 'Amount Invested Monthly']
    age_labels = df_scale_vals['age_category'].unique()

    # get colors, reverse them so dark colors will be at the back of the plot
    color_sequence = px.colors.sequential.Plasma_r[::-1]

    # to make sure sequential colors are not too similar so skip one every time
    skip_interval = 2

    # plot
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

        # add one radar figure per age category
        fig5.add_trace(go.Scatterpolar(
            r=list_radar[i],
            theta=labels,
            fill='toself',
            name=f'{age_labels[i]}',
            text=combined_text_values,
            # Set the color
            marker=dict(color=color)
        ))

    # make adjustments to layout
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
    Output('bar-chart', 'figure', allow_duplicate=True),
    Input('violin-plot', 'clickData')
)
def update_bar_chart(click_data):
    """
    Callback function to update the bar chart based on the selected occupation in the violin plot.

    :param click_data: The click data received from the violin-plot component containing information about the clicked
        point.
    :return: Updated bar chart figure showing the distribution of annual incomes based on the selected occupation
    """
    # nothing clicked on violin chart, return chart with all data
    if click_data is None:
        return fig1
    # get selected occupation from click data
    selected_occupation = click_data['points'][0]['x']
    # prepare dataframe for plotting
    filtered_df = df[df['Occupation'] == selected_occupation]
    income_counts = filtered_df['income_bin'].value_counts().sort_index()

    # plot the bar chart with the dataframe created based on the click data
    fig1_ = px.bar(income_counts, x=income_counts.values, y=income_counts.index, orientation='h',
                   labels={'x': 'Number of Individuals', 'y': 'Income Range'},
                   title='<b>Distribution of Annual Incomes</b>', color_discrete_sequence=['#fb9f3a'])
    # add layout
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


@app.callback(
    Output('violin-plot', 'figure', allow_duplicate=True),
    Input('bar-chart', 'clickData')
)
def update_violin_plot(click_data):
    """
    Callback function to update the violin plot based on the selected occupation in the violin bar chart.

    :param click_data: The click data received from the bar chart component containing information about the clicked
        point.
    :return: Updated violin plot figure showing the distribution of annual incomes by occupation based on data
    """
    # if nothing is clicked, return the original figure
    if click_data is None:
        return fig2

    # prepare dataframe with data from click and filter out five most common occupations
    selected_income_range = click_data['points'][0]['label']
    filtered_df = df[df['income_bin'] == selected_income_range]
    occupation_counts = filtered_df['Occupation'].value_counts()  # count occupations
    top_occupations = occupation_counts.head(5).index.tolist()  # top 5
    filtered_df_top_occupations = filtered_df[filtered_df['Occupation'].isin(top_occupations)]

    # plot the violin chart
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


@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    [Input('bar-chart', 'clickData'),
     Input('violin-plot', 'clickData')]
)
def update_scatter_plot(bar_click, violin_click):
    """
    Callback function to update the scatter plot based on the selected in the violin plot and/or scatter plot

    :param bar_click: The click data received from the bar chart component containing information about the clicked
        point.
    :param violin_click: The click data received from the violin plot component containing information about the
        clicked point
    :return: Updated violin plot figure showing the distribution of annual incomes by occupation based on data
    """
    # access which input triggered the callback
    ctx = dash.callback_context

    # if input triggered the callback, return default figure
    if not ctx.triggered_id:
        fig3.update_traces(opacity=0.5)
        return fig3

    # extract the clicked data
    clicked_input = ctx.triggered_id.split('.')[0]

    # if click comes from bar chart return appropriate figure
    if clicked_input == 'bar-chart' and 'label' in bar_click['points'][0]:
        selected_income_range = bar_click['points'][0]['label']
        # use a condition that captures the selected_income_range
        income_df = df[df['income_bin'] == selected_income_range].copy()
        income_df['Percent_Month_Invest'] = (income_df['Amount_invested_monthly'] / income_df[
            'Monthly_Inhand_Salary']) * 100
        # plot
        fig3_ = px.scatter(
            income_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
            title='<b>Monthly Income vs. Percentage of Monthly Income Invested for</b>',
            labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                    'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
            color_discrete_sequence=['#fb9f3a'])
        fig3_.update_traces(opacity=0.5)
        fig3_.update_layout(title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested</b>',
                            title_x=0.5, height=450, width=675)
        return fig3_

    # if click comes from violin plot return appropriate figure
    elif clicked_input == 'violin-plot' and 'x' in violin_click['points'][0]:
        selected_profession = violin_click['points'][0]['x']
        profession_df = df[df['Occupation'] == selected_profession].copy()
        profession_df['Percent_Month_Invest'] = (profession_df['Amount_invested_monthly'] / profession_df[
            'Monthly_Inhand_Salary']) * 100
        # plot
        fig3_ = px.scatter(
            profession_df, x='Monthly_Inhand_Salary', y='Percent_Month_Invest',
            title=f'<b>Monthly Income vs. Percentage of Monthly Income Invested</b>',
            labels={'Monthly_Inhand_Salary': 'Monthly Inhand Salary (USD)',
                    'Percent_Month_Invest': 'Percent of Monthly Income Invested (%)'},
            color_discrete_sequence=['#fb9f3a'])
        fig3_.update_traces(opacity=0.5)
        fig3_.update_layout(
            title_text='<b>Monthly Income vs. Percentage of Monthly Income Invested}</b>',
            title_x=0.5, height=450, width=675)
        return fig3_


@app.callback(
    Output('radar-chart', 'figure', allow_duplicate=True),
    [Input('scatter-plot', 'selectedData'),
     Input('reset-button', 'n_clicks')]
)
def update_radar_chart(selected_data, n_clicks):
    """
    Callback function to update the radar chart based on the selected in the scatter plot or from the reset button

    :param selected_data: the data that has been selected inside the scatter plot
    :param n_clicks: how many times the reset button has been clicked
    :return: Updated radar chart figure showing multiple attributes per age category
    """
    # access which input triggered the callback
    ctx = dash.callback_context

    # if nothing is triggered return original chart
    if not ctx.triggered_id and n_clicks > 0:
        return create_radar_chart(scale_df)

    # avoids issue with button
    if selected_data is None:
        return create_radar_chart(scale_df)

    # prepare data by creating an adjusted monthly income more appropriate for jobs with an irregular income
    points = selected_data['points']
    x_vals_selected = [point['x'] for point in points]
    radar_df = df[df['Monthly_Inhand_Salary'].isin(x_vals_selected)]
    radar_df['Avg_Monthly_Salary'] = radar_df.Annual_Income / 12

    # to avoid the dashboard not working, return the original figure if the df is empty
    if radar_df.empty:
        return create_radar_chart(scale_df)

    # prepare data by scaling
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
        return create_radar_chart(scale_df)

    # create dataframe to use as input for function
    scaled_values = scaler.fit_transform(df_group[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=[f'SC_{col}' for col in columns_to_scale])
    result_df = pd.concat([df_group, scaled_df], axis=1)
    return create_radar_chart(result_df)


@app.callback(
    Output('bar-chart', 'figure', allow_duplicate=True),
    Output('violin-plot', 'figure', allow_duplicate=True),
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Output('radar-chart', 'figure', allow_duplicate=True),
    Input('reset-button', 'n_clicks')
)
def reset_graphs(n_clicks):
    """
    Function for the reset button, when the button is clicked, all the original graphs are displayed

    :param n_clicks: how many times the button has been clicked
    :return: all the original graphs either by return graph itself, or calling function to create it
    """
    # if the button is not clicked, or it's the first time, return the initial figures
    if n_clicks is None or n_clicks == 0:
        return fig1, fig2, fig3, create_radar_chart(scale_df)

    # button is clicked, return the initial figures (aka reset)
    return fig1, fig2, fig3, create_radar_chart(scale_df)


# layout for the dashboard
# layer 1: title, subtitle and reset button
# layer 2: bar chart and violin plot
# layer 3: scatter plot and radar chart
app.layout = html.Div(style={'backgroundColor': '#111111'}, children=[
    html.Div([
        html.H1(children='The Credit Canvas', style={'textAlign': 'center', 'color': 'white'}),
        html.Div(children='Exploring the credit scene like never before!',
                 style={'textAlign': 'center', 'color': 'white'}),
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
        dcc.Graph(id='radar-chart', figure=create_radar_chart(scale_df))
    ], style={'margin': 'auto', 'display': 'flex', 'justify-content': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=False, port=8058)
