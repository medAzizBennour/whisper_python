import os
import tempfile
import flask
from flask import request
from flask_cors import CORS
import whisper
import json
import spacy
app = flask.Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        language = request.form['language']
        model = request.form['model_size']

      
        audio_model = whisper.load_model(model)

        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, 'temp.wav')

        wav_file = request.files['audio_data']
        wav_file.save(save_path)

        if language == 'english':
            result = audio_model.transcribe(save_path, language='english')
            print(result['text'])
            doc = nlp(result['text'])
            action = ""
            name = ""
            for token in doc:
                if token.pos_ == "VERB":
                    action = token.text
                elif token.pos_ == "NOUN":
                    name = token.text
            
            jsonRes={
                "message":result,
                "action":action,
                "name":name
            }
            json_object = json.dumps(jsonRes, indent=3)
        else:
            result = audio_model.transcribe(save_path)

        return json_object
    else:
        return "This endpoint only processes POST wav blob"
