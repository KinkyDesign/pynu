#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
from ast import literal_eval as make_tuple
from copy import deepcopy
import json
import pickle
import base64
import numpy
import math
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__, static_url_path = "")

def getJsonContentsTrain (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
        variables = dataEntry[0]["values"].keys() 
        variables.sort() 
        datapoints =[]
        target_variable_values = []
        for i in range(len(dataEntry)):
            datapoints.append([])
        for i in range(len(dataEntry)):
            for j in variables:
                if j == predictionFeature:
                    target_variable_values.append(dataEntry[i]["values"].get(j))
                else:
                    datapoints[i].append(dataEntry[i]["values"].get(j))
        variables.remove(predictionFeature)
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
    return variables, datapoints, predictionFeature, target_variable_values, parameters

def getJsonContentsTest (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        rawModel = jsonInput["rawModel"]
        additionalInfo = jsonInput["additionalInfo"]
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
        predictionFeature = additionalInfo[0].get("predictedFeature", None)
        variables = dataEntry[0]["values"].keys() 
        variables.sort() 
        datapoints =[]
        for i in range(len(dataEntry)):
            datapoints.append([])
        for i in range(len(dataEntry)):
            for j in variables:
                datapoints[i].append(dataEntry[i]["values"].get(j))
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
    return variables, datapoints, predictionFeature, rawModel

#RF classification
def rfc_train (variable_values, target_variable_values, max_depth=2, n_estimators=10):
    clf = RandomForestClassifier(
        bootstrap=True, 
        class_weight=None, 
        criterion='gini',
        max_depth = max_depth, 
        max_features='auto', 
        max_leaf_nodes=None,
        min_impurity_decrease=0.0, 
        min_impurity_split=None,
        min_samples_leaf=1, 
        min_samples_split=2,
        min_weight_fraction_leaf=0.0, 
        n_estimators = n_estimators, 
        n_jobs=1,
        oob_score=False, 
        random_state=0, 
        verbose=0, 
        warm_start=False)
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#RF class prediction
def rfc_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#RF class train task
@app.route('/pynu/rfc/train', methods = ['POST'])
def create_task_rfc_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # RF Parameters & defaults: max_depth, n_estimators
    max_depth = parameters.get("max_depth", None)
    if not (max_depth):
        max_depth=2
    n_estimators = parameters.get("n_estimators", None)
    if not (n_estimators):
        n_estimators=10

    encoded = rfc_train(datapoints, target_variable_values, max_depth, n_estimators)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    rf_write = open("rfc_model", "w")
    rf_write.writelines(str(encoded))
    rf_write.close()
    """
    return jsonOutput, 201 

#RF class test task
@app.route('/pynu/rfc/test', methods = ['POST'])
def create_task_rfc_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = rfc_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    rf_write = open("rfc_prediction", "w")
    rf_write.writelines(str(predictionList))
    rf_write.close()
    """
    return jsonOutput, 201 

#RF regression
def rfr_train (variable_values, target_variable_values, max_depth=2, n_estimators=10):
    clf = RandomForestRegressor(
        bootstrap=True, 
        criterion='mse', 
        max_depth=2,
        max_features='auto', 
        max_leaf_nodes=None,
        min_impurity_decrease=0.0, 
        min_impurity_split=None,
        min_samples_leaf=1, 
        min_samples_split=2,
        min_weight_fraction_leaf=0.0, 
        n_estimators=10, 
        n_jobs=1,
        oob_score=False, 
        random_state=0, 
        verbose=0, 
        warm_start=False)
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#RF regression prediction
def rfr_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#RF regression train task
@app.route('/pynu/rfr/train', methods = ['POST'])
def create_task_rfr_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # RF Parameters & defaults: max_depth, n_estimators
    max_depth = parameters.get("max_depth", None)
    if not (max_depth):
        max_depth=2
    n_estimators = parameters.get("n_estimators", None)
    if not (n_estimators):
        n_estimators=10

    encoded = rfr_train(datapoints, target_variable_values, max_depth, n_estimators)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    rf_write = open("rfr_model", "w")
    rf_write.writelines(str(encoded))
    rf_write.close()
    """
    return jsonOutput, 201 

#RF regression test task
@app.route('/pynu/rfr/test', methods = ['POST'])
def create_task_rfr_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = rfr_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    rf_write = open("rfr_prediction", "w")
    rf_write.writelines(str(predictionList))
    rf_write.close()
    """
    return jsonOutput, 201 

#MLP classification
def mlpc_train (variable_values, target_variable_values, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(
        activation='relu', 
        alpha=alpha, 
        batch_size='auto',
        beta_1=0.9, 
        beta_2=0.999, 
        early_stopping=False,
        epsilon=1e-08, 
        hidden_layer_sizes=hidden_layer_sizes, 
        learning_rate='constant',
        learning_rate_init=0.001, 
        max_iter=200, momentum=0.9,
        nesterovs_momentum=True, 
        power_t=0.5, 
        random_state=random_state, 
        shuffle=True,
        solver=solver, 
        tol=0.0001, 
        validation_fraction=0.1, 
        verbose=False,
        warm_start=False)
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#MLP class prediction
def mlpc_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#MLP class train task 
@app.route('/pynu/mlpc/train', methods = ['POST'])
def create_task_mlpc_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # MLP Parameters & defaults: solver, alpha, hidden_layer_sizes, random_state
    solver = parameters.get("solver", None)
    if not (solver):
        solver='lbfgs'
    alpha = parameters.get("alpha", None)
    if not (alpha):
        alpha=1e-5
    hidden_layer_sizes = parameters.get("hidden_layer_sizes", None)
    hidden_layer_sizes = make_tuple(hidden_layer_sizes)
    if not (hidden_layer_sizes):
        hidden_layer_sizes=(5, 2)
    random_state = parameters.get("random_state", None)
    if not (random_state):
        random_state=1

    encoded = mlpc_train(datapoints, target_variable_values, solver, alpha, hidden_layer_sizes, random_state)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    mlpc_write = open("mlpc_model", "w")
    mlpc_write.writelines(str(encoded))
    mlpc_write.close()
    """
    return jsonOutput, 201 

#MLP class test class
@app.route('/pynu/mlpc/test', methods = ['POST'])
def create_task_mlpc_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = mlpc_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    mlpc_write = open("mlpc_prediction", "w")
    mlpc_write.writelines(str(predictionList))
    mlpc_write.close()
    """
    return jsonOutput, 201 

#MLP regression train
def mlpr_train (variable_values, target_variable_values, solver='adam', alpha=0.0001, hidden_layer_sizes=(100, ), random_state=None):
    clf = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, 
        activation='relu', 
        solver=solver, 
        alpha=alpha, 
        batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001, 
        power_t=0.5, 
        max_iter=200, 
        shuffle=True, 
        random_state=random_state, 
        tol=0.0001, 
        verbose=False, 
        warm_start=False, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08)
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#MLP regression prediction
def mlpr_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#MLP regression train task 
@app.route('/pynu/mlpr/train', methods = ['POST'])
def create_task_mlpr_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # MLP Parameters & defaults: solver, alpha, hidden_layer_sizes, random_state
    solver = parameters.get("solver", None)
    if not (solver):
        solver='adam'
    alpha = parameters.get("alpha", None)
    if not (alpha):
        alpha=0.0001
    hidden_layer_sizes = parameters.get("hidden_layer_sizes", None)
    hidden_layer_sizes = make_tuple(hidden_layer_sizes)
    if not (hidden_layer_sizes):
        hidden_layer_sizes=(100, )
    random_state = parameters.get("random_state", None)
    if not (random_state):
        random_state=None

    encoded = mlpr_train(datapoints, target_variable_values, solver, alpha, hidden_layer_sizes, random_state)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    mlpr_write = open("mlpr_model", "w")
    mlpr_write.writelines(str(encoded))
    mlpr_write.close()
    """
    return jsonOutput, 201 

#MLP class test class
@app.route('/pynu/mlpr/test', methods = ['POST'])
def create_task_mlpr_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = mlpr_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    mlpr_write = open("mlpr_prediction", "w")
    mlpr_write.writelines(str(predictionList))
    mlpr_write.close()
    """
    return jsonOutput, 201 

#GB classification
def gbc_train (variable_values, target_variable_values, max_depth=3, n_estimators=100):
    clf = GradientBoostingClassifier(
        loss='deviance', 
        learning_rate=0.1, 
        n_estimators=n_estimators, 
        subsample=1.0, 
        criterion='friedman_mse', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_depth=max_depth, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        init=None, 
        random_state=None, 
        max_features=None, 
        verbose=0, 
        max_leaf_nodes=None, 
        warm_start=False, 
        presort='auto')
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#GB class prediction
def gbc_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#GB class train task
@app.route('/pynu/gbc/train', methods = ['POST'])
def create_task_gbc_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # RF Parameters & defaults: max_depth, n_estimators
    max_depth = parameters.get("max_depth", None)
    if not (max_depth):
        max_depth=3
    n_estimators = parameters.get("n_estimators", None)
    if not (n_estimators):
        n_estimators=100

    encoded = gbc_train(datapoints, target_variable_values, max_depth, n_estimators)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    gb_write = open("gbc_model", "w")
    gb_write.writelines(str(encoded))
    gb_write.close()
    """
    return jsonOutput, 201 

#GB class test task
@app.route('/pynu/gbc/test', methods = ['POST'])
def create_task_gbc_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = gbc_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    gb_write = open("gbc_prediction", "w")
    gb_write.writelines(str(predictionList))
    gb_write.close()
    """
    return jsonOutput, 201 

#GB regression
def gbr_train (variable_values, target_variable_values, max_depth=3, n_estimators=100):
    clf = GradientBoostingRegressor(
        loss='ls', 
        learning_rate=0.1, 
        n_estimators=100, 
        subsample=1.0, 
        criterion='friedman_mse', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_depth=3, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        init=None, 
        random_state=None, 
        max_features=None, 
        alpha=0.9, 
        verbose=0, 
        max_leaf_nodes=None, 
        warm_start=False, 
        presort='auto')
    clf.fit (variable_values, target_variable_values) 
    saveas = pickle.dumps(clf)
    encoded = base64.b64encode(saveas)	
    return encoded

#GB regression prediction
def gbr_test (variables, datapoints, predictionFeature, rawModel):
    decoded = base64.b64decode(rawModel)
    clf2 = pickle.loads(decoded)
    predictionList = []
    for i in range (len(datapoints)):
        temp = clf2.predict([datapoints[i]]) ##[]
        finalPrediction = {predictionFeature:temp[0]}
        predictionList.append(finalPrediction)
    return predictionList

#GB regression train task
@app.route('/pynu/gbr/train', methods = ['POST'])
def create_task_gbr_train():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, target_variable_values, parameters = getJsonContentsTrain(readThis)

    # RF Parameters & defaults: max_depth, n_estimators
    max_depth = parameters.get("max_depth", None)
    if not (max_depth):
        max_depth=3
    n_estimators = parameters.get("n_estimators", None)
    if not (n_estimators):
        n_estimators=100

    encoded = gbr_train(datapoints, target_variable_values, max_depth, n_estimators)
    predictedString = predictionFeature + " predicted"
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictedString}], 
        "independentFeatures": variables, 
        "predictedFeatures": [predictedString] 
    }
    jsonOutput = jsonify( task )
    """
    gr_write = open("gbr_model", "w")
    gr_write.writelines(str(encoded))
    gr_write.close()
    """
    return jsonOutput, 201 

#GB regression test task
@app.route('/pynu/gbr/test', methods = ['POST'])
def create_task_gbr_test():
    if not request.environ['body_copy']:
        abort(500)
    readThis = json.loads(request.environ['body_copy'])
    variables, datapoints, predictionFeature, rawModel = getJsonContentsTest(readThis)
    predictionList = gbr_test(variables, datapoints, predictionFeature, rawModel)
    task = {
        "predictions": predictionList
    }
    jsonOutput = jsonify( task )
    """
    gb_write = open("gbc_prediction", "w")
    gb_write.writelines(str(predictionList))
    gb_write.close()
    """
    return jsonOutput, 201 

class WSGICopyBody(object):
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        from cStringIO import StringIO
        input = environ.get('wsgi.input')
        length = environ.get('CONTENT_LENGTH', '0')
        length = 0 if length == '' else int(length)
        body = ''
        if length == 0:
            environ['body_copy'] = ''
            if input is None:
                return
            if environ.get('HTTP_TRANSFER_ENCODING','0') == 'chunked':
                while (1):
                    temp = input.readline() ## 
                    
                    if not temp:
                        break
                    body +=temp
            size = len(body)
        else:
            body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = StringIO(body)
        app_iter = self.application(environ, 
                                    self._sr_callback(start_response))
        return app_iter

    def _sr_callback(self, start_response):
        def callback(status, headers, exc_info=None):
            start_response(status, headers, exc_info)
        return callback

if __name__ == '__main__': 
    app.wsgi_app = WSGICopyBody(app.wsgi_app) ##
    app.run(host="0.0.0.0", port = 5000, debug = True)	

############################################################
#### Docker
# docker ps
# docker ps -a
# cd .. root docker file
# docker build -t python-rf .
# docker tag python-rf hub.jaqpot.org/python-rf
# U: j P: jh
# docker push hub.jaqpot.org/python-rf
# 
############################################################
#### GB Classification
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainCategorical.json http://localhost:5000/pynu/gbc/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testCategorical.json http://localhost:5000/pynu/gbc/test
#
############################################################
#### GB Regression
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainContinuous.json http://localhost:5000/pynu/gbr/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testContinuous.json http://localhost:5000/pynu/gbr/test
#
############################################################
#### MLP Classification
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainCategorical.json http://localhost:5000/pynu/mlpc/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testCategorical.json http://localhost:5000/pynu/mlpc/test
#
############################################################
#### MLP Regression
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainContinuous.json http://localhost:5000/pynu/mlpr/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testContinuous.json http://localhost:5000/pynu/mlpr/test
#
############################################################
#### RF Classification
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainCategorical.json http://localhost:5000/pynu/rfc/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testCategorical.json http://localhost:5000/pynu/rfc/test
#
############################################################
#### RF Regression
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainContinuous.json http://localhost:5000/pynu/rfr/train
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/testContinuous.json http://localhost:5000/pynu/rfr/test
#
############################################################
#### Image
# curl -i -H "Transfer-encoding:chunked" -H "Content-Type:application/json" -X POST -d @C:/Python27-15/trainCategorical.json http://192.168.99.100:5000/pws/rfc/train
#
############################################################