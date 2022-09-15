import streamlit as st
import pandas as pd
import datetime as dt
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
# project = form.selectbox('project', PROJECT_LIST)
projects = form.multiselect('Projects', PROJECT_LIST)
storey = form.number_input("Storey", min_value=1, max_value=128, value=10, step=1)
size = form.number_input("Size (sqft)", min_value=1, max_value=10000, value=900, step=10)
price = form.number_input("Asking price", min_value=1, max_value=10_000_000, value=500000, step=1000)
generate = form.form_submit_button("Generate")

# =============================================================================

def load_projects(df, project_list):
    pdf = df[ df['full_name'].isin(project_list) ].dropna().sort_values('time', ascending=False)
    pdf['size_sqft'] = pdf['size_sqft'].astype(int)
    return pdf


def print_project_stats(pdf):
    towns = sorted(list(pdf['town'].unique()))
    blocks = sorted(list(pdf['block'].unique()))
    projects = sorted(list(pdf['full_name'].unique()))
    stats = {'Towns': ", ".join(x for x in towns),
             'Projects': ", ".join(x for x in projects),
             'Blocks': ", ".join(x for x in blocks),
             'Sample size': len(pdf),
             'First transaction': pdf['time'].min().strftime("%Y-%m"),
             'Latest transaction': pdf['time'].max().strftime("%Y-%m"),
             }
    return stats


def plot_project_block_avg(pdf):
    vdf = pdf.groupby(['block','size_sqft'])['cost_psf'].mean().unstack().round(2)
    fig, ax = plt.subplots( figsize=(7,5), tight_layout=True)
    sns.heatmap(vdf, cmap='RdYlGn_r', square=True)
    plt.title('Avg Cost PSF by Block', pad=20)
    plt.xlabel('Size (sqft)')
    plt.ylabel('Block')
    plt.show()


def find_days_to_expiry(df):
    # Month of purchase and lease does not always concur
    X = df[['lease_years']]
    Y = df[['days_ago']]
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    res = regr.predict([[0]])[0][0]
    expiry_date = dt.datetime.today() + dt.timedelta(days=res)
    print(df['lease_years'].min(), res, expiry_date)
    return res


def check_project_year(projects):
    years = set([x[-4:] for x in projects])
    if len(years)>1:
        year_list = ", ".join(x for x in years)
        st.text(f"Warning: Projects in {len(years)} different years selected - [{year_list}]\nResults may be inaccurate")

def MLR_predict(df, projects, storey_mid, size_sqft, asking_price, print_stats=False):
    feature_cols = ['lease_years', 'storey_mid', 'size_sqft']
    X = df[feature_cols]
    Y = df[['cost_psf']]

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    if print_stats:
        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)

    df['pred'] = regr.predict(X)
    df['actual'] = Y

    dte = find_days_to_expiry(df)
    project_expiry = dt.datetime.today() + dt.timedelta(days=dte)
    current_lease_years = dte/365.25

    res = regr.predict([[current_lease_years, storey_mid, size_sqft]])
    inputs = {'lease_years': current_lease_years,
              'storey_mid': storey_mid,
              'size_sqft': size_sqft}

    if print_stats:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        print(model.summary())

    cost_psf = res[0][0]
    asking_psf = asking_price / size_sqft

    xlabel_dict = {'lease_years': 'Lease Remaining',
                   'storey_mid': 'Storey',
                   'size_sqft': 'Size (sqft)'}
    i = 1
    fig, ax = plt.subplots(1,3, figsize=(15,5), tight_layout=True)
    for feature in feature_cols:
        ax = plt.subplot(1,3,i)
        sns.regplot( x=df[feature], y=df['cost_psf'], scatter_kws={'s':12}, color='ivory')
        plt.scatter( inputs[feature], cost_psf, s=100, c='dodgerblue', marker='^', zorder=9, label='Fair' )
        plt.scatter( inputs[feature], asking_psf, s=100, c='r', marker='^', zorder=9, label='Asking' )
        plt.grid(linestyle=':', color='grey')
        xmin, xmax = min(df[feature].min(), inputs[feature]), max(df[feature].max(), inputs[feature])
        xrange = xmax-xmin
        adj = xrange*0.05
        plt.xlim([xmin-adj, xmax+adj])
        plt.ylabel('Cost PSF')
        plt.xlabel(xlabel_dict[feature])
        plt.legend()
        if feature=='lease_years':
            ax.invert_xaxis()
        i += 1
    plt.suptitle(", ".join(x for x in projects), fontsize=20)
    plt.savefig("output_image.png", dpi=200, bbox_inches='tight', pad_inches=0.3)
    return cost_psf

# =============================================================================

if generate:
    if len(projects)>0:

        check_project_year(projects)

        pdf = load_projects(df, projects)

        plot_project_block_avg(pdf)

        stats = print_project_stats(pdf)

        pred_cost_psf = MLR_predict(pdf, projects, storey, size, price)

        pred_total_cost = pred_cost_psf*size
        asking_psf = price / size

        output_text = ""
        output_text += f"Projects: {stats['Projects']}"
        output_text += f"\nTowns: {stats['Towns']}"
        output_text += f"\nBlocks: {stats['Blocks']}"
        output_text += "\n"
        output_text += f"\nStorey: {storey}"
        output_text += f"\nSize: {size} sqft"
        output_text += f"\nAsking price: ${price:,.0f}"
        output_text += "\n"
        output_text += f"\nPredicted fair cost PSF: ${pred_cost_psf:.2f}"
        output_text += f"\nPredicted fair price: ${pred_total_cost:,.0f}"
        output_text += f"\nPremium: ${price-pred_total_cost:,.0f} ({price/pred_total_cost-1:.2%})"
        output_text += "\n"
        output_text += f"\nSample size: {stats['Sample size']}"
        output_text += f"\nFirst transaction: {stats['First transaction']}"
        output_text += f"\nLatest transaction: {stats['Latest transaction']}"

        st.text(f"{output_text}")

        st.image('output_image.png', output_format='png')

        if price>2e6:
            st.text(f"Huh ${price/1e6:.1f} million?! You wanna pay so much meh??")
        if size>3000:
            st.text(f"Wa {size} sqft you sure this one HDB or not?")
        if size<200:
            st.text(f"This one square feet not squre metres leh. You sure only {size} sqft?")

    else:
        st.text("You trying to be cute is it? No project how to analyze?")
