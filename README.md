# Detecção de Objetos com YOLOv5

Este projeto treina um modelo YOLOv5 para detectar objetos específicos ('copo' e 'estante') em imagens. O treinamento e a detecção são realizados utilizando um notebook Jupyter no Google Colab.

## Descrição

O objetivo deste projeto é utilizar a arquitetura YOLOv5 para treinar um modelo de detecção de objetos em um dataset customizado contendo imagens com copos e estantes. O notebook automatiza o processo de configuração do ambiente, treinamento do modelo e execução da detecção em imagens de teste.

## Dataset

O dataset utilizado para o treinamento consiste em imagens e anotações para as classes 'copo' e 'estante'. Os dados estão organizados da seguinte forma:
* Imagens de treino e validação localizadas no Google Drive (`/content/drive/MyDrive/2025/images/`).
* Arquivos de anotações (labels) correspondentes no Google Drive (`/content/drive/MyDrive/2025/labels/`).
* Um arquivo de configuração `copo.yaml` define os caminhos para o dataset e as classes.

## Configuração do Ambiente

1.  **Montar Google Drive:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2.  **Clonar Repositório YOLOv5:**
    ```bash
    !git clone [https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)
    ```
3.  **Instalar Dependências:**
    ```bash
    !pip install -r yolov5/requirements.txt
    ```
4.  **Copiar Arquivo de Configuração do Dataset:**
    ```bash
    !cp /content/drive/MyDrive/2025/copo.yaml yolov5/data/
    ```

## Treinamento

O modelo foi treinado utilizando o script `train.py` do YOLOv5. Foram realizados dois treinamentos, um com 40 épocas e outro com 60. O treinamento de 40 épocas apresentou resultados mais equilibrados entre as classes.

* **Comando de Treinamento (40 épocas):**
    ```bash
    !python yolov5/train.py --data copo.yaml --weights yolov5s.pt --img 640 --epochs 40
    ```
* **Pesos Treinados:** Os melhores pesos do treinamento de 40 épocas são salvos em `yolov5/runs/train/exp/weights/best.pt`.

## Detecção (Inferência)

A detecção de objetos em novas imagens é realizada utilizando o script `detect.py`.

* **Comando de Detecção:**
    ```python
    # (Código Python para encontrar a pasta de treino mais recente e rodar a detecção)
    import os
    import subprocess

    def get_latest_train_run_folder():
        subfolders = [f.path for f in os.scandir('yolov5/runs/train') if f.is_dir()]
        latest_folder = max(subfolders, key=os.path.getctime, default=None)
        return latest_folder

    latest_run = get_latest_train_run_folder() # Assume que 'exp' é a pasta do treino de 40 épocas
    if latest_run:
        result = subprocess.run(f'python yolov5/detect.py --weights {latest_run}/weights/best.pt --img 640 --source /content/drive/MyDrive/2025/images/test --data yolov5/data/copo.yaml', shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    else:
        print("Não foi possível encontrar a pasta de treinamento mais recente.")

    ```
* **Resultados da Detecção:** As imagens com as detecções são salvas no diretório `yolov5/runs/detect/exp<N>/` (onde `<N>` é o número do experimento, por exemplo, `exp2`).

## Análise dos Resultados

* **Modelo de 40 Épocas:** Apresentou bom desempenho geral, com mAP50 próximo de 100% e mAP50-95 de 69.7% nos dados de validação iniciais, indicando boa capacidade de identificação, embora com potencial para melhorar a precisão da localização. Este modelo parece mais equilibrado.
* **Modelo de 60 Épocas:** Mostrou melhora na detecção da classe 'copo' (mAP50-95 de 85.2%), mas piorou significativamente na detecção da classe 'estante' (mAP50-95 de 32.2%), resultando em um mAP50-95 geral inferior (58.7%).

A análise sugere que o modelo treinado por 40 épocas oferece um melhor equilíbrio geral para as classes detectadas neste dataset.

## Dependências Principais

* Python 3
* PyTorch
* OpenCV
* Bibliotecas listadas em `yolov5/requirements.txt`
