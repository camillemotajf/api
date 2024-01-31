from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import tensorflow as tf
from keras import backend as K
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Carregar o modelo treinado a partir do arquivo .h5
SEED = 179


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


@app.route("/previsao", methods=["POST"])
def prever():
    model = load_model("modelo_Lag5.h5")
    scaler_carregado = joblib.load("meu_scaler_minmax.joblib")
    # Receber os dados da requisição em formato JSON
    dados_entrada = request.get_json()

    # Verificar se os dados de entrada estão corretos
    if "valores" not in dados_entrada:
        return jsonify({"erro": "Os valores de entrada não foram fornecidos"}), 400

    # Extrair os valores de entrada
    valores_entrada = dados_entrada["valores"]
    valores_entrada = np.array(valores_entrada).reshape(-1, 1)
    valores_entrada = scaler_carregado.transform(valores_entrada)
    # Verificar se há 5 valores de entrada
    if len(valores_entrada) != 5:
        return jsonify({"erro": "Deve haver exatamente 5 valores de entrada"}), 400

    entrada_array = np.array([valores_entrada])

    previsao = model.predict(entrada_array)

    previsao_desnorm = scaler_carregado.inverse_transform(previsao).tolist()

    # Retornar a previsão em formato JSON
    return jsonify({"previsao": previsao_desnorm})


@app.route("/retreinamento", methods=["POST"])
def retreinar():
    count = 0
    set_global_determinism(seed=SEED)
    data = pd.read_csv(
        "dados_diarios_14Nov_outliers.csv"
    )  # CSV q representa a requisição ao servidor dos dados todos
    data = data.rename(
        columns={"Daily Power yields (kWh)": "energy", "inverterdatetime": "time"}
    )
    data_power = data["energy"]
    data_power = np.array(data_power)
    scaler = MinMaxScaler()
    data_power = data_power.reshape(-1, 1)
    normalized_data = scaler.fit_transform(data_power)

    data_power = pd.Series(
        [value for row in normalized_data for value in row], name="energy"
    )
    X = pd.concat(
        [
            data_power.shift(1),
            data_power.shift(2),
            data_power.shift(3),
            data_power.shift(4),
            data_power.shift(5),
        ],
        axis=1,
    )
    y = pd.concat([data_power.shift(-5)], axis=1)
    X.dropna(inplace=True)
    y.dropna(subset=["energy"], inplace=True)
    X = X.to_numpy()
    y = y.to_numpy()
    y = y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, stratify=None
    )
    model = tf.keras.models.load_model("australia_grid_5.h5")
    model.trainable = False
    print(model.summary())
    for layer in model.layers:
        layer.trainable = False
    base_inputs = model.layers[0].input
    base_outputs = model.layers[-2].output

    output = layers.Dense(1, name="output")(base_outputs)
    new_model = tf.keras.Model(inputs=base_inputs, outputs=output)
    new_model.summary()
    new_model.compile(
        optimizer="adam", loss="mean_squared_error"
    )  # Note que estou usando o "adam" logo ta com lr no default

    history = new_model.fit(
        X_train, y_train, epochs=20, batch_size=32, validation_split=0.3
    )

    predicoes = new_model.predict(X_test)

    predicoes = scaler.inverse_transform(predicoes)

    real = scaler.inverse_transform(y_test.reshape(-1, 1))

    erro = (np.mean(((predicoes - real) ** 2))) ** 0.5

    if count == 0:
        ultimo_loss_lido = 124

    # Salvar o modelo treinado
    if erro < ultimo_loss_lido:
        new_model.save("seu_modelo_retreinado_API.h5")
        ultimo_loss_lido = erro
        return jsonify(
            {
                "Ribeirão": {
                    "mensagem": "Modelo retreinado com sucesso e o erro caiu em relação ao anterior e será atualizado no BD",
                    "valor_Loss_retreinamento": history.history["loss"][-1],
                    "erro_Anterior": ultimo_loss_lido,
                    "erro_Atual": erro,
                }
            }
        )
    else:
        model.save("seu_modelo_retreinado_API.h5")
        ultimo_loss_lido = erro
        return jsonify(
            {
                "Ribeirão": {
                    "mensagem": "Modelo retreinado com sucesso, mas o erro não caiu em relação ao anterior. Logo, será mantido o anterior no BD",
                    "valor_Loss_retreinamento": history.history["loss"][-1],
                    "erro_Anterior": ultimo_loss_lido,
                    "erro_Atual": erro,
                }
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
