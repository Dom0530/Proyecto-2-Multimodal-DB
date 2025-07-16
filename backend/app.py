from flask import Flask
from routes import api_blueprint
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DevelopmentConfig 

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
app.register_blueprint(api_blueprint)

if __name__ == "__main__":
    print(f"Mensaje: {app.config['CUSTOM_MESSAGE']}")
    app.run(port=5000)
