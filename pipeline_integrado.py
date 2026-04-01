import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import faiss
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Usamos esto para que no tire error si no estás en Colab, pero funcione bien si lo estás
try:
    from google.colab.patches import cv2_imshow
    en_colab = True
except ImportError:
    en_colab = False

# =====================================================================
# FASE 1: CONTEO Y DETECCIÓN (YOLO)
# =====================================================================
class DetectorVacas:
    def __init__(self, modelo_path='yolov8n.pt'):
        self.modelo = YOLO(modelo_path)

    def detectar(self, frame):
        # Para fotos sueltas, no necesitamos 'track', solo 'predict'
        resultados = self.modelo.predict(frame, classes=[19], verbose=False)
        vacas_detectadas = []

        if resultados and resultados[0].boxes and resultados[0].boxes.xyxy is not None:
            boxes = resultados[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                # Como son fotos sueltas sin seguimiento temporal, el ID es genérico
                vacas_detectadas.append({"track_id": i, "bbox": xyxy})

        return vacas_detectadas

# =====================================================================
# FASE 2: IDENTIFICACIÓN INDIVIDUAL (PyTorch + FAISS + Albumentations)
# =====================================================================
class RedEmbeddings(nn.Module):
    def __init__(self, dimension_salida=256, ruta_pesos=None):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, dimension_salida)
        )

        if ruta_pesos and os.path.exists(ruta_pesos):
            self.load_state_dict(torch.load(ruta_pesos, map_location=torch.device('cpu')))
            print(f"✅ Cerebro experto cargado desde: {ruta_pesos}")
        elif ruta_pesos:
            print(f"⚠️ ATENCIÓN: No se encontró el archivo {ruta_pesos}. Usando cerebro de fábrica.")

    def forward(self, x):
        caracteristicas = self.backbone(x)
        embedding = self.head(caracteristicas)
        return nn.functional.normalize(embedding, p=2, dim=1)

class IdentificadorVacas:
    def __init__(self, ruta_pesos=None, umbral_similitud=0.68):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo = RedEmbeddings(dimension_salida=256, ruta_pesos=ruta_pesos).to(self.device)
        self.modelo.eval()
        self.umbral = umbral_similitud

        self.transformacion = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.indice_faiss = faiss.IndexFlatIP(256)
        self.nombres_vacas = []

    def generar_embedding(self, imagen_recortada):
        rgb = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2RGB)
        tensor = self.transformacion(image=rgb)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.modelo(tensor).cpu().numpy().astype(np.float32)
        return embedding

    def registrar_vaca(self, id_vaca, embedding):
        self.indice_faiss.add(embedding)
        self.nombres_vacas.append(id_vaca)

    def identificar_vaca(self, embedding):
        if self.indice_faiss.ntotal == 0:
            return "Desconocida", 0.0

        distancias, indices = self.indice_faiss.search(embedding, 1)
        mejor_puntaje = float(distancias[0][0])
        indice_ganador = int(indices[0][0])

        if mejor_puntaje >= self.umbral:
            return self.nombres_vacas[indice_ganador], mejor_puntaje
        else:
            return "Desconocida", mejor_puntaje

# =====================================================================
# FASE 3: LÓGICA DE NEGOCIO
# =====================================================================
def generar_reporte_y_alertas(vacas_vistas_hoy, stock_esperado, lista_vacas_registradas):
    print("\n" + "="*40)
    print("--- REPORTE DE STOCK DIARIO ---")
    conteo_hoy = len(vacas_vistas_hoy)
    print(f"Total de vacas únicas detectadas: {conteo_hoy} / {stock_esperado}")

    identificadas = [v for v in vacas_vistas_hoy if v != "Desconocida"]
    faltantes = set(lista_vacas_registradas) - set(identificadas)

    if faltantes:
        print(f"⚠️ ALERTA DE IDENTIDAD: No se vio a las vacas: {faltantes}")
    else:
        print("✅ Todo el rodeo base está presente e identificado.")
    print("="*40 + "\n")

# =====================================================================
# EJECUCIÓN PRINCIPAL (AHORA CON CARPETAS)
# =====================================================================
if __name__ == "__main__":
    print("Iniciando herramientas...")
    detector = DetectorVacas('yolov8n.pt')
    
    # 1. Conectamos el archivo de Ramiro (Asegúrate de que la ruta sea correcta tras hacer git clone)
    ruta_modelo_ramiro = "outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt"
    identificador = IdentificadorVacas(ruta_pesos=ruta_modelo_ramiro)

    # 2. ENROLAMIENTO SIMULADO
    print("Cargando base de datos FAISS...")
    identificador.registrar_vaca("Vaca_A", np.random.rand(1, 256).astype(np.float32))
    identificador.registrar_vaca("Vaca_B", np.random.rand(1, 256).astype(np.float32))
    lista_rodeo_oficial = ["Vaca_A", "Vaca_B"]

    # 3. PROCESAR CARPETA DEL DATASET
    # Aquí pones la ruta donde está la carpeta de imágenes que subió Ramiro
    ruta_dataset = "ruta/a/la/carpeta/del/dataset/OpenCows2020" 
    
    # Busca todas las fotos .jpg y .png en esa carpeta
    imagenes_paths = glob.glob(os.path.join(ruta_dataset, "*.jpg")) + glob.glob(os.path.join(ruta_dataset, "*.png"))

    vacas_identificadas_en_dataset = set()
    ultimo_frame_dibujado = None

    if not imagenes_paths:
        print(f"\n❌ ERROR: No se encontraron imágenes en la carpeta '{ruta_dataset}'.")
        print("Revisa la ruta de la carpeta y vuelve a intentar.")
    else:
        print(f"\n📷 Se encontraron {len(imagenes_paths)} imágenes. Iniciando análisis...\n")

        for img_path in imagenes_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # FASE 1: Detectar
            detecciones = detector.detectar(frame)

            for det in detecciones:
                x1, y1, x2, y2 = det["bbox"]
                recorte = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                if recorte.size == 0: continue

                # FASE 2: Identificar
                emb = identificador.generar_embedding(recorte)
                id_vaca, confianza = identificador.identificar_vaca(emb)
                
                # Guardamos la identidad encontrada
                vacas_identificadas_en_dataset.add(id_vaca)

                # Dibujar cuadraditos y texto
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, id_vaca, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            ultimo_frame_dibujado = frame.copy()

        # Mostrar la última imagen analizada como prueba de que funcionó
        print("\n¡Análisis del dataset terminado! Aquí tienes la última foto analizada:")
        if ultimo_frame_dibujado is not None and en_colab:
            cv2_imshow(ultimo_frame_dibujado)
        elif ultimo_frame_dibujado is not None:
            # Si lo corres en tu PC local y no en colab
            cv2.imshow("Prueba Dataset", ultimo_frame_dibujado)
            cv2.waitKey(0)

        # FASE 3: Ejecutar Alertas
        generar_reporte_y_alertas(vacas_identificadas_en_dataset, stock_esperado=2, lista_vacas_registradas=lista_rodeo_oficial)
