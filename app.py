import flask
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
with open(f'model/MLmodelworldflags.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
        if flask.request.method == 'GET':
            return (flask.render_template('main.html'))
        if flask.request.method == 'POST':
            language = flask.request.form['language']
            area = flask.request.form['area']
            population = flask.request.form['population']
            landmass = flask.request.form['landmass']
            zone = flask.request.form['zone']
            input_variables = pd.DataFrame([[landmass, zone, area, population, language]],
                                           columns=['landmass', 'zone', 'area', 'population', 'language'],
                                           dtype=float)
            prediction = model.predict(input_variables)[0]
            return flask.render_template('main.html',
                                         original_input={
                                             'Landmass': landmass,
                                             'Zone': zone,
                                             'Area': area,
                                            'Population': population,
                                             'Language': language
                                                         },
                                         result=prediction,
                                         )

if __name__ == '__main__':
    app.run()