# Cool Websites

## 🤖 Build a Robot

Open `index.html` in a browser for an interactive guide to building your first robot. It covers:

- **Robot lab:** move the head and arm servos with sliders, and see the serial commands the brain sends
- **Anatomy:** what each part does (brain, spinal cord, muscles, senses, heart)
- **Memory & RAM:** a hands-on demo of RAM versus storage
- **Shopping list**, **8 build steps** and **safety rules**

The real robot code is in [`robot/`](robot/README.md): a Python brain with memory for the Raspberry Pi, and Arduino firmware for the limbs. You can try the brain right now without any hardware:

```bash
cd robot/brain && python3 brain.py
```
