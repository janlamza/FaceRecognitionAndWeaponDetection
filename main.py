from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import face_recognition
import cv2
import numpy as np
import base64
import os
import asyncio
import time
import psycopg2
from datetime import datetime
from pathlib import Path
import uuid
from collections import deque
from typing import Optional, List, Dict, Tuple, Deque
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# Inicijalizacija FastAPI aplikacije i montiranje statičkih datoteka
# -----------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# Inicijalizacija baze podataka i paralelnog izvršavanja
# -----------------------------------------------------------------------------

# Povezivanje s PostgreSQL bazom putem psycopg2
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)

# Thread pool za paralelno računanje face embeddinga (4 CPU jezgre)
enc_executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# Parametri i memorija za prepoznavanje lica
# -----------------------------------------------------------------------------

FACE_RECOGNITION_INTERVAL: float = 0.2        # Minimalni razmak (sekunde) između pokušaja prepoznavanja
UNKNOWN_FACE_MATCH_THRESHOLD: float = 0.6     # Prag sličnosti za praćenje nepoznatih osoba

face_busy = False                             # Oznaka da je obrada lica u tijeku
known_face_encodings: List[np.ndarray] = []   # Encodinzi poznatih osoba iz baze
known_face_metadata: List[Dict] = []          # Prateći metapodaci za poznata lica

latest_face_boxes: List[Dict] = []            # Detektirani boxovi lica za overlay
face_history: Deque[Dict] = deque(maxlen=15)  # Povijest nedavnih prepoznavanja za prikaz

# -----------------------------------------------------------------------------
# Parametri i memorija za detekciju oružja
# -----------------------------------------------------------------------------

YOLO_INTERVAL: float = 0.2                    # Minimalni razmak (sekunde) između YOLO detekcija
YOLO_CONFIDENCE_THRESHOLD: float = 0.8        # Minimalna pouzdanost detekcije da bi se prihvatila
WEAPON_WATCHED_CLASSES = {"pistol", "rifle", "knife", "gun"}  # Klase koje se promatraju
MAX_WEAPON_COUNT = 4                          # Maksimalan broj boxova koje se prikazuje

yolo_model = YOLO("models/yolo12-weapon.pt")  # Učitani YOLO model za detekciju oružja
yolo_busy = False                             # Oznaka da je YOLO detekcija u tijeku
latest_weapon_boxes: List[Dict] = []          # Detektirani boxovi oružja za overlay


# -----------------------------------------------------------------------------
# Učitava poznate osobe iz baze i priprema encodirane vektore za prepoznavanje
# -----------------------------------------------------------------------------
def load_known_faces_from_db() -> None:
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, image_path, name, surname, age, nationality, criminal_record
            FROM persons
        """)
        for row in cursor.fetchall():
            person_id, image_path, name, surname, age, nationality, criminal_record = row

            if not os.path.exists(image_path):
                print(f"[WARN] Slika ne postoji: {image_path}")
                continue

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                print(f"[WARN] Nema detektiranog lica u slici: {image_path}")
                continue

            known_face_encodings.append(encodings[0])
            known_face_metadata.append({
                "id": person_id,
                "name": name,
                "surname": surname,
                "age": age,
                "nationality": nationality,
                "criminal_record": criminal_record,
            })

# Poziv funkcije odmah pri pokretanju aplikacije
load_known_faces_from_db()

# -----------------------------------------------------------------------------
# Pretvara base64 Data-URI (npr. "data:image/jpeg;base64,....") u BGR sliku (ndarray)
# Vraća: NumPy sliku ili None ako dekodiranje ne uspije.
# -----------------------------------------------------------------------------
def get_img(data_uri: str) -> Optional[np.ndarray]:
    """
    1) Odreže meta-dio prije zareza („data:image/jpeg;base64,”).
    2) Base64-dekodira ostatak u niz bajtova.
    3) Pretvori bajtove u NumPy vektor i OpenCV-om dekodira u BGR sliku.
    """
    try:
        header, b64_data = data_uri.split(",", 1)      # "data:..." | "base64..."
        nparr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
        if nparr.size == 0:                            # zaštita od praznog dekodiranja
            raise ValueError("prazan niz bajtova")
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # BGR format
    except Exception as exc:                           # binascii.Error, ValueError, cv2.error …
        print(f"[WARN] get_img() decoding failed: {exc}")
        return None
        

# -----------------------------------------------------------------------------
# Provjerava je li bounding box geometrijski ispravan:
# - odbacuje preuske (npr. linije) ili preširoke detekcije.
# Argumenti su koordinate bounding boxa.
# Vraća False ako je omjer visina/širina ekstreman.
# -----------------------------------------------------------------------------
def is_valid_box(left: int, top: int, right: int, bottom: int) -> bool:
    w = right - left
    h = bottom - top
    ratio = h / w if w else 0.0

    if ratio > 5 or ratio < 0.2:
        return False
    return True


# -----------------------------------------------------------------------------
# Prepoznaje lica na slici, vodi evidenciju poznatih / nepoznatih i priprema
#     face_boxes  –  boxovi za overlay
#     new_captures –  izrezi novootkrivenih lica za galeriju
# Funkcija je optimizirana za brzinu:
#   1)  detektira sva lica (HOG/CNN ovisno o face_recognition konfigu)
#   2)  kontrolira učestalost obrade (FACE_RECOGNITION_INTERVAL)
#   3)  računa embeddinge paralelno u ThreadPool-u (enc_executor)
#   4)  uspoređuje s poznatima; nepoznate grupira i prati
#   5)  sprema rezultate u globalne strukture (face_boxes, face_history …)
# -----------------------------------------------------------------------------
def recognize_faces(
    img: np.ndarray,
    seen_people: set,
    seen_unknown_encodings: List[np.ndarray],
    last_recognition_time: float
) -> Tuple[List[Dict], List[Dict], float]:

    face_boxes: List[Dict] = []
    new_captures: List[Dict] = []

    # ------------------------------------------------------------------
    # 1) Detekcija lokacija lica
    # ------------------------------------------------------------------
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)

    # ------------------------------------------------------------------
    # 2) FPS kontrola – preskače ako je obrada prečesta ili nema lica
    # ------------------------------------------------------------------
    now              = time.time()
    do_recognition   = (now - last_recognition_time) > FACE_RECOGNITION_INTERVAL
    if not (do_recognition and face_locations):
        return face_boxes, [], last_recognition_time

    # ------------------------------------------------------------------
    # 3) Paralelno računanje embeddinga (jedan future po licu)
    # ------------------------------------------------------------------
    futures = [
        enc_executor.submit(
            face_recognition.face_encodings,
            rgb_img,
            [location]               # ← svaki location: (top, right, bottom, left)
        )
        for location in face_locations
    ]

    # Pairing location ↔ embedding, zadržavamo redoslijed i izbacujemo prazne
    paired = [
        (loc, embedding[0])          # embedding lista → uzmi prvi vektor
        for loc, embedding in zip(face_locations, (f.result() for f in futures))
        if embedding
    ]

    # ------------------------------------------------------------------
    # 4) Obrada svakog lica: validacija boxa, usporedba, bilježenje
    # ------------------------------------------------------------------
    for (top, right, bottom, left), face_enc in paired:

        # Geometrijska provjera boxa
        if not is_valid_box(left, top, right, bottom):
            continue

        # ----- osnovni meta -----------------------------
        meta = {
            "name": "Unknown",
            "surname": "",
            "age": "",
            "nationality": "",
            "criminal_record": ""
        }
        person_id = None

        # ------------------------------------------------------
        # A) Usporedba s poznatim licima
        # ------------------------------------------------------
        if known_face_encodings:
            matches   = face_recognition.compare_faces(known_face_encodings, face_enc)
            distances = face_recognition.face_distance(known_face_encodings, face_enc)
            best_idx  = int(np.argmin(distances))
            if matches[best_idx]:
                meta      = known_face_metadata[best_idx]
                person_id = meta["id"]

        # ------------------------------------------------------
        # B) Ako lice još nije poznato – grupiraj ga među "unknown"
        # ------------------------------------------------------
        if person_id is None:
            for idx, unk_enc in enumerate(seen_unknown_encodings):
                if np.linalg.norm(unk_enc - face_enc) < UNKNOWN_FACE_MATCH_THRESHOLD:
                    person_id  = f"unknown {idx}"
                    meta["name"] = person_id
                    break
            else:
                seen_unknown_encodings.append(face_enc)
                person_id  = f"unknown {len(seen_unknown_encodings) - 1}"
                meta["name"] = person_id

        # ------------------------------------------------------
        # C) Galerija – novo lice pojavi se prvi put
        # ------------------------------------------------------
        if person_id not in seen_people:
            seen_people.add(person_id)
            crop = img[top:bottom, left:right]
            _, jpeg = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(jpeg.tobytes()).decode()

            new_captures.append({
                "id": meta.get("id"),
                "name": meta["name"],
                "surname": meta["surname"],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": f"data:image/jpeg;base64,{b64_img}",
                "criminal_record": meta.get("criminal_record", "")
            })

        # ------------------------------------------------------
        # D) Spremi box za overlay
        # ------------------------------------------------------
        face_boxes.append({
            "top":          top,
            "right":        right,
            "bottom":       bottom,
            "left":         left,
            "id":           meta.get("id"),
            "name":         meta.get("name", "Unknown"),
            "surname":      meta.get("surname", ""),
            "age":          meta.get("age", ""),
            "nationality":  meta.get("nationality", ""),
            "criminal_record": meta.get("criminal_record", "")
        })

    # ------------------------------------------------------------------
    # 5) Vraćamo rezultat + ažurirani timestamp
    # ------------------------------------------------------------------
    return face_boxes, new_captures, now


# --------------------------------------------------------------------------------
# Funkcija za detekciju oružja
# Koristi YOLO model s filtriranjem po nazivu klase, pouzdanosti i geometriji boxa.
# Broj detekcija ograničen s MAX_WEAPON_COUNT.
# Vraća listu bounding‑boxova oružja za overlay.
# --------------------------------------------------------------------------------

def detect_weapon_boxes(img: np.ndarray) -> List[Dict]:
    boxes: List[Dict] = []

    for result in yolo_model(img, verbose=False, max_det=MAX_WEAPON_COUNT):
        for box in result.boxes:
            cls = int(box.cls[0])                      # numpy.int64 → int
            name = str(yolo_model.names[cls])          # osiguraj string
            confidence = float(box.conf[0])            # numpy.float32 → float

            if name not in WEAPON_WATCHED_CLASSES or confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue
                
            if len(boxes) >= MAX_WEAPON_COUNT:
                return boxes

            left,  top, right, bottom = map(int, box.xyxy[0])

            if not is_valid_box(left, top, right, bottom):
                continue

            boxes.append(
                {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                    "label": name,
                    "confidence": confidence,
                    "weapon": True,
                }
            )

    return boxes

# -----------------------------------------------------------------------------
# Root endpoint (GET "/") – vraća početnu HTML stranicu iz datoteke 'index.html'
# Koristi se za prikaz korisničkog sučelja u pregledniku.
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# -----------------------------------------------------------------------------
# WebSocket endpoint  (ws://host:port/ws)
# • Prima Base64-encoded frameove s klijenta (script.js)       →  get_img()
# • Lansira dva paralelna zadatka:
#     1) prepoznavanje lica   (recognize_faces)  – u thread-poolu
#     2) detekciju oružja     (detect_weapon_boxes) – u thread-poolu,
#        ali rjeđe, prema YOLO_INTERVAL
# • Po završetku zadataka sprema globalne rezultate i šalje JSON klijentu.
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    # --- lokalni "state" po klijentu ----------------------------------------
    last_recognition_time: float = 0.0   # za kontrolu učestalosti face-recogna
    last_yolo_time:        float = 0.0   # za kontrolu učestalosti YOLO-a
    seen_people: set       = set()       # ID-jevi lica koja smo već prikazali
    seen_unknown_encodings: List[np.ndarray] = []  # encodinzi "unknown" lica

    global face_busy, yolo_busy, latest_face_boxes, latest_weapon_boxes

    try:
        while True:
            # ----------------------------------------------------------------
            # 1) Primi i dekodiraj sliku
            # ----------------------------------------------------------------
            data_uri = await websocket.receive_text()
            img = get_img(data_uri)
            if img is None:                          # loš frame → samo pauziraj
                await asyncio.sleep(0.01)
                continue

            loop = asyncio.get_running_loop()

            # ----------------------------------------------------------------
            # 2) Lansiraj zadatak prepoznavanja lica (ako nije već u tijeku)
            # ----------------------------------------------------------------
            if not face_busy:
                face_busy = True

                def face_task():
                    # kopija slike jer ju prosljeđujemo u background-thread
                    return recognize_faces(
                        img.copy(),
                        seen_people,
                        seen_unknown_encodings,
                        last_recognition_time
                    )

                def face_done(fut):                  # callback na završetak
                    nonlocal last_recognition_time
                    global face_busy, latest_face_boxes

                    face_busy = False
                    boxes, caps, last = fut.result()
                    last_recognition_time = last
                    latest_face_boxes = boxes
                    face_history.extend(caps)

                loop.run_in_executor(None, face_task).add_done_callback(face_done)

            # ----------------------------------------------------------------
            # 3) Lansiraj YOLO (detekciju oružja) rjeđe – prema YOLO_INTERVAL
            # ----------------------------------------------------------------
            now = time.time()
            if (now - last_yolo_time > YOLO_INTERVAL) and not yolo_busy:
                yolo_busy = True
                last_yolo_time = now

                def yolo_task():
                    return detect_weapon_boxes(img.copy())

                def yolo_done(fut):
                    global yolo_busy, latest_weapon_boxes
                    yolo_busy = False
                    latest_weapon_boxes = fut.result()

                loop.run_in_executor(None, yolo_task).add_done_callback(yolo_done)

            # ----------------------------------------------------------------
            # 4) Pošalji klijentu trenutno dostupne rezultate
            # ----------------------------------------------------------------
            await websocket.send_json({
                "faces":        [f["name"] for f in latest_face_boxes],
                "face_boxes":   latest_face_boxes,
                "face_captures": list(face_history),
                "weapon_boxes": latest_weapon_boxes,
            })

            await asyncio.sleep(0.01)  # lagani "yield" da ne blokiramo petlju

    except WebSocketDisconnect:
        print("[INFO] Client disconnected")


# -----------------------------------------------------------------------------
# REST endpoint: POST /save_unknown_person
# Sprema nepoznatu osobu iz forme (ime, slika, opcionalni podaci).
# • Sprema sliku u known_faces/
# • Upisuje osobu u bazu (tablica persons)
# • Dodaje encoding i metapodatke u memoriju (poznata lica)
# -----------------------------------------------------------------------------
@app.post("/save_unknown_person")
async def save_unknown_person(
    name: str = Form(...),
    surname: str = Form(""),
    age: Optional[int] = Form(None),
    nationality: str = Form(""),
    criminal_record: str = Form(""),
    image_data: str = Form(...),
):
    try:
        # ---------------------------------------------------------------
        # 1) Dekodiraj i spremi sliku na disk (known_faces/<uuid>.jpg)
        # ---------------------------------------------------------------
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        filename = f"{uuid.uuid4().hex}.jpg"
        path = Path("known_faces") / filename
        path.write_bytes(img_bytes)

        # ---------------------------------------------------------------
        # 2) Upis u bazu: kreiraj novu osobu
        # ---------------------------------------------------------------
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO persons (image_path, name, surname, age, nationality, criminal_record)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (str(path), name, surname, age, nationality, criminal_record),
            )
            new_person_id = cursor.fetchone()[0]
            conn.commit()

        # ---------------------------------------------------------------
        # 3) Izračunaj encoding slike i dodaj u memoriju
        # ---------------------------------------------------------------
        image = face_recognition.load_image_file(str(path))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_metadata.append({
                "id": new_person_id,
                "name": name,
                "surname": surname,
                "age": age,
                "nationality": nationality,
                "criminal_record": criminal_record,
            })
        else:
            print(f"[WARN] Novo lice nije prepoznato u spremljenoj slici: {path}")

        return {"status": "success", "id": new_person_id}

    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
        
# -----------------------------------------------------------------------------
# REST endpoint: GET /get_person/{person_id}
# Dohvaća podatke o osobi iz baze na temelju ID-a.
# Ako je osoba pronađena, vraća strukturirani JSON s podacima.
# Ako nije pronađena, vraća error poruku.
# -----------------------------------------------------------------------------
@app.get("/get_person/{person_id}")
async def get_person(person_id: int):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, name, surname, age, nationality, criminal_record
            FROM persons
            WHERE id = %s
            """,
            (person_id,),
        )
        row = cursor.fetchone()

    if row:
        # Redak iz baze mapiramo na ključeve kao dictionary
        keys = ("id", "name", "surname", "age", "nationality", "criminal_record")
        return dict(zip(keys, row))  # type: ignore[arg-type]

    # Ako nije pronađeno ništa – vraćamo poruku o grešci
    return {"error": "Person not found"}


# -----------------------------------------------------------------------------
# REST endpoint: GET /person_image/{person_id}
# Dohvaća putanju do slike osobe iz baze podataka prema zadanom ID-u.
# Ako slika postoji, vraća je kao JPEG datoteku.
# Ako slika ne postoji u bazi ili na disku, vraća poruku o grešci.
# -----------------------------------------------------------------------------
@app.get("/person_image/{person_id}")
async def get_person_image(person_id: int):
    # 1. Dohvati putanju slike iz baze
    with conn.cursor() as cursor:
        cursor.execute("SELECT image_path FROM persons WHERE id = %s", (person_id,))
        row = cursor.fetchone()

    if not row:
        return {"error": "Slika nije pronađena u bazi"}

    image_path = row[0]

    # 2. Provjeri postoji li datoteka na disku
    if not os.path.exists(image_path):
        return {"error": "Datoteka ne postoji na disku"}

    # 3. Vrati sliku kao JPEG datoteku
    return FileResponse(image_path, media_type="image/jpeg")

# -----------------------------------------------------------------------------
# REST endpoint: POST /update_person
# Ažurira podatke o osobi u bazi podataka na temelju zadanog ID-a.
# Također ažurira podatke u memoriji (known_face_metadata) na temelju ID-a.
# -----------------------------------------------------------------------------
@app.post("/update_person")
async def update_person(
    id: int = Form(...),
    name: str = Form(...),
    surname: str = Form(...),
    age: int = Form(...),
    nationality: Optional[str] = Form(None),
    criminal_record: Optional[str] = Form(None),
):
    with conn.cursor() as cursor:
        # 1. Provjera postoji li osoba s tim ID-em
        cursor.execute("SELECT 1 FROM persons WHERE id = %s", (id,))
        if not cursor.fetchone():
            return {"status": "error", "message": "Osoba nije pronađena u bazi"}

        # 2. Ažuriraj zapis u bazi podataka
        cursor.execute("""
            UPDATE persons
            SET name = %s, surname = %s, age = %s, nationality = %s, criminal_record = %s
            WHERE id = %s
        """, (name, surname, age, nationality, criminal_record, id))
        conn.commit()

    # 3. Ažuriraj podatke u memoriji (na temelju ID-a)
    for meta in known_face_metadata:
        if meta.get("id") == id:
            meta["name"] = name
            meta["surname"] = surname
            meta["age"] = age
            meta["nationality"] = nationality
            meta["criminal_record"] = criminal_record
            break

    return {"status": "updated"}
    
    
    
