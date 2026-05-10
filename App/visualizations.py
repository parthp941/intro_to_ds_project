import matplotlib.pyplot as plt
import streamlit as st


def plot_line_chart(title, x_values, y_values, y_label):
    chart = plt.figure(figsize=(12, 6))
    for label, values in y_values.items():
        plt.plot(x_values, values, label=label)
    plt.title(title)
    plt.xlabel("Dates")
    plt.ylabel(y_label)
    plt.legend()
    st.pyplot(chart)
    plt.close()


def plot_histogram(title, values, x_label):
    chart = plt.figure(figsize=(12, 6))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    st.pyplot(chart)
    plt.close()


def plot_bar_chart(title, x_values, y_values, y_label):
    chart = plt.figure(figsize=(12, 6))
    plt.bar(x_values, y_values)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(chart)
    plt.close()
