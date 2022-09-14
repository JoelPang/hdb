import streamlit as st
import pandas as pd
import glob
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
plt.rcParams["figure.dpi"] = 80
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Montserrat']

# =============================================================================
# HDB Fair Price Calculator - JP 2021
# =============================================================================

st.set_page_config(page_icon="🏙️", page_title="HDB Fair Value Calculator")
st.header("""**HDB Fair Value Calculator**""")


file_list = glob.glob("./data/*.csv")
df = pd.concat([pd.read_csv(file) for file in file_list])
df['time'] = pd.to_datetime( df['time'] )

PROJECT_LIST = sorted(df['full_name'].unique())

form = st.form(key="submit-form")
project = form.selectbox('project', PROJECT_LIST)
storey = form.number_input("storey", min_value=1, max_value=600, value=10, step=1)
size = form.number_input("size", min_value=1, max_value=10000, value=1000, step=10)
price = form.number_input("Asking price", min_value=1, max_value=10_000_000, value=500000, step=1000)
generate = form.form_submit_button("Generate")

# =============================================================================

def load_project(df, project):
    pdf = df[ df['full_name']==project ].dropna().sort_values('time', ascending=False)
    pdf['size_sqft'] = pdf['size_sqft'].astype(int)
    return pdf

def plot_project_block_avg(pdf):
    vdf = pdf.groupby(['block','size_sqft'])['cost_psf'].mean().unstack().round(2)
    fig, ax = plt.subplots( figsize=(7,5), tight_layout=True)
    sns.heatmap(vdf, cmap='RdYlGn_r', square=True)
    plt.title('Avg Cost PSF by Block', pad=20)
    plt.xlabel('Size (sqft)')
    plt.ylabel('Block')
    plt.show()

def MLR_predict(df, group_name, storey_mid, size_sqft, asking_price, print_stats=False):
    feature_cols = ['lease_years', 'days_ago', 'storey_mid', 'size_sqft']
    X = df[feature_cols]
    Y = df[['cost_psf']]

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    if print_stats:
        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)

    df['pred'] = regr.predict(X)
    df['actual'] = Y

    lease_years = df['lease_years'].min() # current lease remaining
    res = regr.predict([[lease_years, 0, storey_mid, size_sqft]])
    inputs = {'lease_years': lease_years,
              'days_ago': 0,
              'storey_mid': storey_mid,
              'size_sqft': size_sqft}

    if print_stats:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        print(model.summary())

    cost_psf = res[0][0]
    asking_psf = asking_price / size_sqft

    xlabel_dict = {'lease_years': 'Lease Remaining',
                   'days_ago': 'Days Ago',
                   'storey_mid': 'Storey',
                   'size_sqft': 'Size (sqft)'}
    i = 1
    fig, ax = plt.subplots(1,4, figsize=(18,5), tight_layout=True)
    for feature in feature_cols:
        ax = plt.subplot(1,4,i)
        sns.regplot( x=df[feature], y=df['cost_psf'], scatter_kws={'s':12}, color='wheat')
        plt.scatter( inputs[feature], cost_psf, s=100, c='dodgerblue', marker='^', zorder=9, label='Fair' )
        plt.scatter( inputs[feature], asking_psf, s=100, c='r', marker='^', zorder=9, label='Asking' )
        plt.grid(linestyle=':', color='grey')
        xmin, xmax = df[feature].min(), df[feature].max()
        xrange = xmax-xmin
        adj = xrange*0.05
        plt.xlim([xmin-adj, xmax+adj])
        plt.ylabel('Cost PSF')
        plt.xlabel(xlabel_dict[feature])
        if feature=='lease_years':
            ax.invert_xaxis()
        i += 1
    plt.suptitle(project, fontsize=20)
    plt.savefig("output_image.png", dpi=140, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    return cost_psf

# =============================================================================

if generate:
    pdf = load_project(df, project)
    plot_project_block_avg(pdf)

    pred_cost_psf = MLR_predict(pdf, project, storey, size, price)

    pred_total_cost = pred_cost_psf*size
    asking_psf = price / size

    output_text = ""
    output_text += f"Project: {project}"
    output_text += f"\nStorey: {storey}"
    output_text += f"\nSize: {size} sqft"
    output_text += f"\nAsking price: ${price:,.0f}"
    output_text += "\n"
    output_text += f"\nPredicted fair cost PSF: ${pred_cost_psf:.2f}"
    output_text += f"\nPredicted fair price: ${pred_total_cost:,.0f}"
    output_text += f"\nPremium: ${price-pred_total_cost:,.0f} ({price/pred_total_cost-1:.2%})"

    st.text(output_text)

    st.image('output_image.png', output_format='png')
