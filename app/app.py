from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from model import VectorSpaceModel

model = VectorSpaceModel()
app = Flask(__name__)

@app.route('/')
def getHome():
  return render_template('home.html')

@app.route('/query', methods=['GET', 'POST'])
def getQueryResults():
  if request.method == 'GET':
    return redirect('/')
  query = request.form.get('query')
  documents = model.executeQuery(query)
  return render_template('query.html', documents=documents, query=query)