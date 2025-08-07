from flask import Flask, render_template
from data import predict_result_RF

app = Flask(__name__)

@app.route('/')
def KNN():
    result_RF = predict_result_RF()
    return render_template('index.html', final_resultRF=result_RF)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
