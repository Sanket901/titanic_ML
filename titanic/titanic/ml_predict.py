def prediction_model(pclass,sex,age,sibsp,parched,fare,embarked,title):
    import pickle
    x=[[pclass,sex,age,sibsp,parched,fare,embarked,title]]
    randomforest=pickle.load(open('titanic_model.sav','rb'))
    prediction=randomforest.predict(x)
    if prediction==0:
        prediction="Not Survived"
    elif prediction==1:
        prediction="Surrvived"
    else:
        prediction="Error"
    return prediction
