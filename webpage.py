from flask import Flask, render_template, Response, url_for
from webcam import WebCam


app = Flask(__name__)

path_cascade = './resources/haarcascade_frontalface_default.xml'
path_model = './resources/models/cls.pth'
tolerance = 10
features = ['Attractive', 'Bags_Under_Eyes', 'Bangs', 'Chubby', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Young']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect/<int:resolution>')
def detect(resolution):
    title = f'Multi-Label Classification - {resolution}p'
    return render_template('detect.html', resolution=resolution, title=title)

@app.route('/webcam/<int:resolution>')
def webcam(resolution):
    if resolution == 1080:
        resolution_ = (1920, 1080)
    elif resolution == 720:
        resolution_ = (1280, 720)
    elif resolution == 480:
        resolution_ = (640, 480)
    elif resolution == 360:
        resolution_ = (480, 360)
    cam = WebCam(features, path_cascade, path_model, tolerance=tolerance, resolution=resolution_)
    return Response(cam.feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False) # set debug to false for hosting
