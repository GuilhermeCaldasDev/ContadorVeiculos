import cv2
from ultralytics import YOLO
import yt_dlp
import numpy as np
from collections import defaultdict
import folium
from folium.plugins import HeatMap
import webbrowser
import time

def abrir_video():
    print("Abrindo stream do YouTube...")
    video_url = 'https://youtu.be/YDYVz-4_2Wo'
    # video_url = 'https://youtu.be/6dp-bvQ7RWo'

    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_url = info.get('url', None)

    if video_url is None:
        print("Não foi possível extrair a URL do vídeo.")
        return

    cap = cv2.VideoCapture(video_url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.")
        return

    model = YOLO('yolov8n.pt')
    contador_veiculos = 0
    linha_deteccao_y = 500  # Coordenada Y da linha de detecção

    # Dicionário para armazenar a trajetória dos carros
    track_history = defaultdict(list)
    passou_da_linha = defaultdict(lambda: False)

    tempo_maximo = 60
    inicio = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Não foi possível capturar o frame.")
            break

        results = model.track(source=frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False, tracker="bytetrack.yaml", persist=True)

        # Inicializa a variável de desenho com o frame atual
        desenho_com_centro = frame.copy()

        # Desenha a linha de detecção
        cv2.line(desenho_com_centro, (500, 300), (520, 800), (255, 255, 255), 5)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                centro = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                car_id = int(box.id[0]) if box.id is not None else None

                if car_id is not None:
                    # Adiciona o ponto central atual à trajetória do carro
                    track_history[car_id].append(centro)

                    if len(track_history[car_id]) > 30:  # Manter apenas os últimos 30 pontos
                        track_history[car_id].pop(0)

                    # Desenha a trajetória do carro
                    points = np.array(track_history[car_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(desenho_com_centro, [points], isClosed=False, color=(0, 0, 255), thickness=3)

                    # Verifica se o carro cruzou a linha de baixo para cima
                    if len(track_history[car_id]) > 1:
                        if track_history[car_id][-2][1] > linha_deteccao_y and centro[1] <= linha_deteccao_y:
                            if not passou_da_linha[car_id]:
                                passou_da_linha[car_id] = True
                                contador_veiculos += 1
                                cv2.line(desenho_com_centro, (500, 300), (520, 800), (0, 0, 255), 5)
                                print("Carro contabilizado")
                        else:
                            passou_da_linha[car_id] = False

                # blankImage[y_min:y_max, x_min:x_max] += 1
                desenho_boxes = cv2.rectangle(desenho_com_centro, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                desenho_boxes_com_centro = cv2.circle(desenho_boxes, centro, 5, (255, 0, 0), 2)

        # Desenha o contador de veículos na tela
        cv2.putText(desenho_com_centro, f"Total carros: {contador_veiculos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if time.time() - inicio > tempo_maximo:
            print("Tempo máximo atingido, encerrando o loop.")
            heatMap(contador_veiculos)
            break

        cv2.imshow('Janela', desenho_com_centro)
        # cv2.imshow('HeatMap', imgFinal)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # heatMap(contador_veiculos)
    cap.release()
    cv2.destroyAllWindows()

def heatMap(contador_veiculos):
    locations = {
        'BeiraMarGol': {'coords': [-27.6023705295809, -48.61720844131336], 'count': contador_veiculos},
        'NewFitness': {'coords': [-22.91174433157462, -43.17817169728206], 'count': 0},
        'Top Residencial': {'coords': [-22.926544003551204, -43.34908926844538], 'count': 0},
        'Aquarius': {'coords': [-22.932121038206493, -43.33410022370146], 'count': 0},
    }

    # Localização inicial do mapa
    Brasil = [-15.788497, -47.899873]

    # Cria o mapa
    baseMap = folium.Map(
        width='100%',
        height='100%',
        location=Brasil,
        zoom_start=4
    )

    # Preparar dados para o mapa de calor
    heat_data = [[data['coords'][0], data['coords'][1], data['count']] for data in locations.values()]

    # Adicionar marcadores com títulos
    for location, data in locations.items():
        folium.Marker(
            location=data['coords'],
            popup=f"{location}: {data['count']} detections",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(baseMap)

    # Adicionar o mapa de calor ao mapa base
    HeatMap(heat_data, radius=15).add_to(baseMap)

    # Salva o mapa em um arquivo HTML
    map_filename = 'map.html'
    baseMap.save(map_filename)

    # Abre o arquivo HTML no navegador padrão
    webbrowser.open(map_filename)

if __name__ == '__main__':
    abrir_video()