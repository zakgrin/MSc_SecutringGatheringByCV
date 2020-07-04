from flask import Flask, render_template
from utilities import database
import os

app = Flask(__name__)

# Take top words from database
os.chdir(os.path.dirname(__file__))
path = os.path.join(os.getcwd(), "words.db")
top_10_list = database.print_database_table(database_path=path)


@app.route('/')
def home():
    return render_template('home.html', top_list=top_10_list)


if __name__ == '__main__':
    app.run()
