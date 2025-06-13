import cv2
import numpy as np
import time
import winsound
import os
import sys
import re
import pyttsx3
from collections import defaultdict, Counter

# Intento de importar mediapipe
try:
    import mediapipe as mp
except ImportError:
    print("Error: el paquete 'mediapipe' no está instalado. Por favor instálelo e intente de nuevo.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh

# Constantes
DWELL_TIME = 1.0
KEY_SIZE = 80
KEY_MARGIN = 15
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
KEYBOARD_AREA_HEIGHT = 300
CAMERA_WIDTH = 240
CAMERA_HEIGHT = 180

# Teclado
KEYBOARD = [
    ['1','2','3','4','5','6','7','8','9','0','Borrar'],
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L','Ñ','Shift'],
    ['Z','X','C','V','B','N','M',',','.','"'],
    ['SPACE']
]
LEFT_IRIS_INDICES  = [468,469,470,471,472]
RIGHT_IRIS_INDICES = [473,474,475,476,477]

class WordPredictor:
    def __init__(self, path_txt, ngram=2):
        self.ngram = ngram
        self.model = defaultdict(Counter)
        self.unigram = Counter()
        self._build_model(path_txt)

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)

    def _build_model(self, path_txt):
        if not os.path.exists(path_txt):
            return
        text = open(path_txt, encoding='utf-8').read()
        tokens = self._tokenize(text)
        self.unigram.update(tokens)
        for i in range(len(tokens)-(self.ngram-1)):
            key = tuple(tokens[i:i+self.ngram-1])
            self.model[key][tokens[i+self.ngram-1]] += 1

    def predict(self, prefix, top_k=3):
        toks = self._tokenize(prefix)
        key = tuple(toks[-(self.ngram-1):]) if len(toks)>=self.ngram-1 else tuple(toks)
        counter = self.model[key]
        if counter:
            return [w for w,_ in counter.most_common(top_k)]
        last = toks[-1] if toks else ''
        candidates = [(w,c) for w,c in self.unigram.items() if w.startswith(last)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [w for w,_ in candidates[:top_k]]


class EyeTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.calibration_stage = 0
        self.calibration_start = 0
        self.calibration_points = []
        self.calibration_complete = False
        self.cal_min = (0,0)
        self.cal_max = (1,1)
        self.last_cursor = (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        self.smoothing_window_size = 10
        self.cursor_history = [self.last_cursor]*self.smoothing_window_size

    def get_iris_center(self, lm, idx):
        pts = np.array([[lm[i].x, lm[i].y] for i in idx])
        return pts.mean(axis=0)

    def calibrate(self, frame, gv):
        t = time.time()
        if self.calibration_stage == 0:
            self.calibration_stage = 1
            self.calibration_start = t
        targets = [
            (SCREEN_WIDTH//4, SCREEN_HEIGHT//4),
            (3*SCREEN_WIDTH//4, SCREEN_HEIGHT//4),
            (SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4),
            (3*SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4)
        ]
        x,y = targets[self.calibration_stage-1]
        cv2.circle(frame, (x,y), 20, (0,0,255), -1)
        if t - self.calibration_start > DWELL_TIME:
            self.calibration_points.append(tuple(gv))
            self.calibration_stage += 1
            self.calibration_start = t
            winsound.Beep(500,100)
            if self.calibration_stage > 4:
                xs = [p[0] for p in self.calibration_points]
                ys = [p[1] for p in self.calibration_points]
                self.cal_min = (min(xs), min(ys))
                self.cal_max = (max(xs), max(ys))
                self.calibration_complete = True
                winsound.Beep(1000,300)

    def detect_gaze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        cursor = self.last_cursor
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            ir_l = self.get_iris_center(lm, LEFT_IRIS_INDICES)
            ir_r = self.get_iris_center(lm, RIGHT_IRIS_INDICES)
            gv = (ir_l + ir_r) / 2
            if not self.calibration_complete:
                self.calibrate(frame, gv)
            else:
                nx = (gv[0] - self.cal_min[0])/(self.cal_max[0]-self.cal_min[0])
                ny = (gv[1] - self.cal_min[1])/(self.cal_max[1]-self.cal_min[1])
                x = int((1-nx)*SCREEN_WIDTH)
                y = int(ny*SCREEN_HEIGHT)
                self.cursor_history.append((x,y))
                if len(self.cursor_history) > self.smoothing_window_size:
                    self.cursor_history.pop(0)
                sm = np.mean(self.cursor_history, axis=0).astype(int)
                cursor = (max(0,min(sm[0],SCREEN_WIDTH-1)),
                          max(0,min(sm[1],SCREEN_HEIGHT-1)))
                self.last_cursor = cursor
        return cursor


class VirtualKeyboard:
    def __init__(self):
        self.text = ""
        self.keys = self._create_keys()
        self.shift = False
        self.hover = None
        self.hover_t = 0
        self.hover_sugg = None
        self.hover_sugg_t = 0
        self.block = False
        self.last_press = 0
        self.last_key = None               # <--- guardamos la última tecla
        self.suggestions = []
        if not os.path.exists('palabras.txt'):
            with open('palabras.txt','w',encoding='utf-8') as f:
                f.write('HOLA\nMUNDO\nPYTHON\n')
        self.words = [l.strip().upper() for l in open('palabras.txt',encoding='utf-8') if l.strip()]
        self.predictor = WordPredictor('corpus.txt', ngram=2)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate',150)
        self.engine.setProperty('volume',1)
        for v in self.engine.getProperty('voices'):
            if 'spanish' in v.name.lower():
                self.engine.setProperty('voice', v.id)
                break

    def _create_keys(self):
        ks = []
        oy = (SCREEN_HEIGHT - KEYBOARD_AREA_HEIGHT)//2
        for r,row in enumerate(KEYBOARD):
            tw = len(row)*KEY_SIZE + (len(row)-1)*KEY_MARGIN
            ox = (SCREEN_WIDTH - tw)//2
            y  = oy + r*(KEY_SIZE+KEY_MARGIN)
            for c,ch in enumerate(row):
                # Si es Borrar, ancho doble
                w = KEY_SIZE*2 if ch=='Borrar' else KEY_SIZE
                ks.append({
                    'char': ch,
                    'pos': (ox + c*(KEY_SIZE+KEY_MARGIN), y),
                    'size': (w, KEY_SIZE),
                    'cd': 0
                })
        return ks


    def update_cd(self):
        for k in self.keys:
            if k['cd']>0: k['cd']-=1
        if self.block and time.time()-self.last_press > 0.2:
            self.block = False

    def get_key(self, cur):
        if cur is None: return None
        for k in self.keys:
            x,y = k['pos']; w,h = k['size']
            if x<=cur[0]<=x+w and y<=cur[1]<=y+h:
                return k
        return None

    def press(self, k):
        if not k or k['cd']>0 or self.block: return
        winsound.Beep(440,100)
        ch = k['char']
        if ch == 'Shift':
            self.shift = not self.shift
        elif ch == 'SPACE':
            self.text += ' '
        elif ch == 'Borrar':
            self.text = self.text[:-1]
        else:
            self.text += ch.upper() if self.shift else ch.lower()
            self.shift = False
            # decir en voz alta
            self.engine.say(ch)
            self.engine.runAndWait()

        # marcar última tecla y cooldown
        self.last_key   = k
        self.block      = True
        self.last_press = time.time()
        k['cd']        = 15

        # actualizar sugerencias
        cands = self.predictor.predict(self.text.lower(), top_k=3)
        self.suggestions = [w.upper() for w in cands]

    def update_hover(self, cur):
        now = time.time()
        # dwell en teclas
        k = self.get_key(cur)
        if k != self.hover:
            self.hover   = k
            self.hover_t = now
        elif k and now - self.hover_t > DWELL_TIME:
            self.press(k)
            self.hover = None

        # dwell en sugerencias (igual coords que en draw_keyboard)
        keyboard_top = (SCREEN_HEIGHT - KEYBOARD_AREA_HEIGHT)//2
        suggestion_y = keyboard_top - 60
        sugg = None
        for i,s in enumerate(self.suggestions):
            x = 50 + i*200; y = suggestion_y; w,h = 180,40
            if x<=cur[0]<=x+w and y<=cur[1]<=y+h:
                sugg = s
                break

        if sugg != self.hover_sugg:
            self.hover_sugg   = sugg
            self.hover_sugg_t = now
        elif sugg and now - self.hover_sugg_t > DWELL_TIME:
            winsound.Beep(880,100)
            parts = self.text.rstrip().split(' ')
            parts[-1] = sugg.lower()
            self.text = ' '.join(parts) + ' '
            self.suggestions = []
            self.hover_sugg  = None
            self.block        = True
            self.last_press   = now


def draw_keyboard(frm, vk):
    frm[:]=0
    for k in vk.keys:
        x,y = k['pos']; w,h = k['size']
        w0 = w*4 if k['char']=='SPACE' else w
        x0 = x - (w0-w)//2
        # solo la última tecla va en verde
        if k is vk.last_key:
            col = (0,255,0)
        elif k['cd']>0:
            col = (0,150,150)
        else:
            col = (200,200,200)
        if k['char']=='Shift' and vk.shift:
            col = (255,200,0)
        cv2.rectangle(frm,(x0,y),(x0+w0,y+h),col,-1)
        cv2.rectangle(frm,(x0,y),(x0+w0,y+h),(50,50,50),2)
        txt = ' ' if k['char']=='SPACE' else k['char']
        ts  = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        cv2.putText(frm,txt,(x0+(w0-ts[0])//2,y+(h+ts[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    # sugerencias
    keyboard_top = (SCREEN_HEIGHT - KEYBOARD_AREA_HEIGHT)//2
    suggestion_y = keyboard_top - 60
    for i,s in enumerate(vk.suggestions):
        x=50+i*200; y=suggestion_y; w,h=180,40
        cv2.rectangle(frm,(x,y),(x+w,y+h),(0,150,255),-1)
        cv2.putText(frm,s,(x+10,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    assert cap.isOpened(), "Error cámara"

    eye = EyeTracker()
    vk  = VirtualKeyboard()

    cv2.namedWindow('Eye Typing System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Eye Typing System',
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    while True:
        ret, cam = cap.read()
        if not ret: break

        cursor = eye.detect_gaze(cam)
        if not eye.calibration_complete:
            cv2.putText(cam,f"Calibrando {eye.calibration_stage}/4",
                        (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('Eye Typing System',cam)
            if cv2.waitKey(5)&0xFF==27: break
            continue

        cursor = eye.detect_gaze(cam)
        canvas = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3),dtype=np.uint8)

        draw_keyboard(canvas, vk)
        cv2.circle(canvas,(int(cursor[0]),int(cursor[1])),10,(0,255,0),-1)
        small = cv2.resize(cam,(CAMERA_WIDTH,CAMERA_HEIGHT))
        canvas[10:10+CAMERA_HEIGHT,
               SCREEN_WIDTH-CAMERA_WIDTH-10:SCREEN_WIDTH-10] = small

        vk.update_cd()
        vk.update_hover(cursor)

        cv2.putText(canvas, vk.text, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow('Eye Typing System', canvas)
        if cv2.waitKey(5)&0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
