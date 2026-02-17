FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xvfb x11-utils libxrender1 libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENV DISPLAY=:99
CMD ["python3", "final13.py"]
