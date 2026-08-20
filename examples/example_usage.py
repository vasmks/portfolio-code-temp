from toontra import Toontra

image_paths = ["page_01.png", "page_02.png", "page_03.png"]

toontra = Toontra()
results = toontra.process(image_paths)

for page in results:
    cleaned_image = page.cleaned

    for bubble in page.bubbles:
        box = bubble.detection.box.as_tuple()
        text = bubble.recognition.text if bubble.recognition else None
        translation = bubble.translation
