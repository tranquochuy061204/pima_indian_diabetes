from flask import Flask, render_template, request
import numpy as np
import pickle
from indian_diabetes import Node, predict_single

# Load mô hình cây quyết định đã lưu
with open("tree_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Dự đoán đầu ra
def predict_single(input_data, tree):
    node = tree
    while not node.is_leaf_node():
        if input_data[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu người dùng nhập
            input_data = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
            ]

            # Dự đoán kết quả
            result = predict_single(np.array(input_data), model)
            prediction = "Có tiểu đường" if result == 1 else "Không có tiểu đường"
        except:
            prediction = "Dữ liệu không hợp lệ. Vui lòng nhập lại."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
