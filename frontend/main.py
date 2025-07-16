from dearpygui.dearpygui import *
import requests

def fetch_data():
    try:
        response = requests.get("http://127.0.0.1:5000/api/data")
        data = response.json()
        set_value("response_text", data["message"])
    except Exception as e:
        set_value("response_text", f"Error: {e}")

with window(label="Aplicación Nativa con Flask", width=400, height=200):
    add_text("Presiona el botón para llamar al backend")
    add_button(label="Llamar API", callback=fetch_data)
    add_text("", tag="response_text")

start_dearpygui()
