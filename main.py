from flask import Flask, jsonify, request
import os
import pickle


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "API para prediccion"


@app.route('/api/v1/predict', methods=['GET'])
def prediction():
    
    tv = int(request.args['TV'])
    radio = int(request.args['radio'])
    newspaper = int(request.args['newspaper'])

    #Cargamos el modelo

    loaded_model = pickle.load(open('./model.pkl', 'rb'))

    new_data = [tv, radio, newspaper]

    #Realizamos la preduccion
    prediction_model = loaded_model.predict([new_data])
    

    #Devolvemos la prediccion

    return jsonify({"repository":prediction_model[0]})
    
if __name__ == '__main__':
    app.run(debug = True, port=5000)
