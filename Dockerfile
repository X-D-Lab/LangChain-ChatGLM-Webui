FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
RUN apt-get update && \
    apt-get install --no-install-recommends -y git python3 python3-dev python3-pip build-essential libmagic-dev poppler-utils tesseract-ocr libreoffice vim libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN pip install --upgrade pip
RUN pip install torch PyPDF2 pdfrw unstructured accelerate
RUN pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2"
COPY . .
RUN pip install -r requirements.txt
CMD ["python3", ""app.py"]
