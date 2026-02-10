import pandas as pd
import streamlit as st
# import matplotlib.pyplot as mtp
import plotly.express as plx
# import numpy as np

url = r"C:\Users\DELL\PycharmProjects\PythonProject\.venv\revision\global_cars_dataset_synthetic.csv"
data = pd.read_csv(url)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", 20)

print("data read")
print(data)

print("data metadataa:")
print(data.info())
print(data.describe())
print(data.shape)
print("-------------------------------")

if True in data.duplicated():
    print("yes, duplication is in the data.")
    data = data.drop_duplicates()
data = data.dropna()

print("Data cleaned")
print(data)
# st.dataframe(data)

data = (data.set_index('Car_ID'))
print(data)
# st.dataframe(data)


datcol = data.columns.values
# dict = {}

# print(datacol)
brands = data["Brand"].value_counts().reset_index()
print(brands)
# st.dataframe(brands)


# single car data
carBrandGroup = data.groupby("Brand")
print(carBrandGroup.get_group("BMW"))
# st.dataframe(carBrandGroup.get_group("BMW"))

# design type
st.write("Table and chart")
body = carBrandGroup[["Brand", "Body_Type"]].value_counts().reset_index()
st.dataframe(body)
fig = plx.bar(body, x="Brand", y="count", color="Body_Type")
fig.update_layout(barmode="group")
st.plotly_chart(fig, use_container_width=True)
# print(body)

fuel = carBrandGroup[["Brand", "Fuel_Type"]].value_counts().reset_index()
st.dataframe(fuel)
fig = plx.bar(fuel, x="Brand", y="count", color="Fuel_Type")
fig.update_layout(barmode="stack")
st.plotly_chart(fig, use_container_width=True)

print(carBrandGroup[["Brand", "Transmission"]].value_counts())
Transmission = carBrandGroup[["Brand", "Transmission"]].value_counts().reset_index()
st.dataframe(Transmission)
fig = plx.bar(Transmission, x="Brand", y="count", color="Transmission")
# fig1 = plx.bar(Transmission, x="Brand", y="count", color="Transmission")
fig.update_layout(barmode="stack")
# fig1.update_layout(barmode="relative")
st.plotly_chart(fig, key="stack", use_container_width=True)
# st.plotly_chart(fig1, key="relative", use_container_width=True)


# st.dataframe(carBrandGroup[["Brand", "Fuel_Type"]].value_counts())
# st.dataframe(carBrandGroup[["Brand", "Transmission"]].value_counts())
# fig, ax = mtp.subplots()
# ax.pie(body["count"], labels=body["Brand"])
# st.pyplot(fig)

# different counts on different basis
print(carBrandGroup[["Brand", "Body_Type", "Fuel_Type"]].value_counts())
# st.dataframe(carBrandGroup[["Brand", "Body_Type", "Fuel_Type"]].value_counts())
whole_chart = carBrandGroup[["Brand", "Body_Type", "Fuel_Type", "Transmission"]].value_counts().reset_index()
st.dataframe(whole_chart)
fig = plx.bar(whole_chart, x="Brand", y="count", color="Body_Type")
fig.update_layout(barmode="group")
st.plotly_chart(fig, use_container_width=True)
# st.dataframe(carBrandGroup[["Brand", "Body_Type", "Fuel_Type", "Transmission"]].value_counts())

# # brand by manufacturing country
# print(carBrandGroup[["Brand", "Manufacturing_Country"]].value_counts())
# # st.dataframe(carBrandGroup[["Brand", "Manufacturing_Country"]].value_counts())
#
# bodyT = data["Body_Type"].value_counts()
# fuelT = data["Fuel_Type"].value_counts()
# tranT = data["Transmission"].value_counts()
# manfactCoun = data["Manufacturing_Country"].value_counts()

# st.dataframe(brands)
click = ""
def clickResult(parm):
    click = ""
    for i in parm["Body_Type"]:
        sel = st.button(i)
        # st.write(sel)
        if sel:
            click = i
    st.write(click)
    return click

for i in brands["Brand"]:
    sel = st.button(i)
    # st.write(sel)
    if sel:
        click = i


if click:
    st.write(click)
    carData = carBrandGroup.get_group(click)
    st.dataframe(carData)
    bodyT = carData["Body_Type"].value_counts().reset_index()
    clickb = ""
    for i in bodyT["Body_Type"]:
        sel = st.button(i)
        # st.write(sel)
        if sel:
            st.write(i)
            clickb = i
    if clickb:
        st.write(clickb)
    # fuelT = carData["Fuel_Type"].value_counts()
    # tranT = carData["Transmission"].value_counts()
    # manfactCoun = carData["Manufacturing_Country"].value_counts()

# st.bar_chart()
