from model_files.image_trans import img_convert
from keras.models import load_model
import pickle

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
##from selenium.webdriver.common.keys import Keys
import sys
import json


app = FastAPI()
leaf = load_model('ex3_rice_model.h5')


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post('/cari_tokped')
# async def tokped(info : Request):
#     req_info = await info.json()
    

@app.post('/predict/')
async def predict(info : Request):
    req_info = await info.json()

    # img64 = requests.get_json()
    img64 = req_info['data_64']
    # print(img64)
    predict_data = img_convert(img64)
    prediction = leaf.predict(predict_data
    )
    label_binarizer = pickle.load(open(r'ex3_rice_transform.pkl', 'rb'))
    ret_data = label_binarizer.inverse_transform(prediction)[0]
    
    result = {
        'predict' : ret_data
    }

    return result


# @app.route('/getmsg/', methods=['GET'])
# def respond():
#     name = request.args.get("name", None)

#     print(f'received: {name}')

#     response = {}

#     if not name:
#         response["ERROR"] = "No name found"
    
#     elif str(name).isdigit():
#         response["ERROR"] = "The name can't be numeric"

#     else:
#         response["MESSAGE"] = f'Welcome {name} to API'

#     return jsonify(response)

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=9696)