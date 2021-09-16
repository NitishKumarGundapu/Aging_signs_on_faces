from keras.models import model_from_json

json_file=open(r"models/dark_s.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("models/dark_s.h5")
print("yes")