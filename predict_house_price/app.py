import flask
from flask import Flask, request, render_template
import numpy as np
import joblib





app = Flask(__name__)


# load ml model
model = joblib.load('model.pkl')


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		
		file = request.form.getlist('input_example[]')
		results = list(map(int, file))
		
		prediction = model.predict([results,])
		
		label = np.squeeze(prediction)
		labell = np.round(label , 3)
		return render_template('index.html', label='{} ريال '.format(labell))


if __name__ == '__main__':
	

	# start 
	app.run(debug=True)
