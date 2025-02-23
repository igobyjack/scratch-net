# Python
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import use  # Uses functions from [use.py](/c:/Desktop/CS projects/scratch-net/use.py)

class DrawingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.resizable(0, 0)
        self.canvas_size = 280  # Use a larger canvas for drawing
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=4, pady=2, sticky=tk.W+tk.E)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # PIL image for drawing and later processing
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image1)
        
        self.button_predict = tk.Button(self, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.grid(row=1, column=1)
        self.label_result = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.label_result.grid(row=2, column=0, columnspan=4)
        
        # Load the trained model parameters
        self.W1, self.b1, self.W2, self.b2 = use.load_model()

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill='white')
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image1)

    def predict(self):
        # Downscale the drawn image to 28x28 pixels using the LANCZOS filter
        img = self.image1.resize((28, 28), Image.Resampling.LANCZOS)
        # Invert colors: model likely expects a dark digit on a white background
        img = ImageOps.invert(img)
        img_np = np.array(img).astype('float32') / 255.0
        X_example = img_np.reshape(28*28, 1)

        # Get predictions using forward propagation from use.py
        _, _, _, A2 = use.forward_prop(self.W1, self.b1, self.W2, self.b2, X_example)
        prediction = use.get_predictions(A2)[0]
        self.label_result.config(text=f"Prediction: {prediction}")

if __name__ == "__main__":
    app = DrawingApp()
    app.mainloop()