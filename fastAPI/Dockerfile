FROM python:3.8-slim

# Install system dependencies for PyTorch, torchaudio, and other packages
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libomp-dev \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your application files into the container
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies directly into the system Python environment
RUN pip install fastapi[all]
RUN pip install opencv-python-headless
RUN pip install torch==1.8.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install torchaudio==0.8.0 -f https://pytorch.org/whl/torch_stable.html
RUN pip install gitpython==3.1.30
RUN pip install matplotlib==3.3
RUN pip install numpy==1.23.5
RUN pip install opencv-python>=4.1.1
RUN pip install pillow>=10.3.0
RUN pip install psutil
RUN pip install PyYAML>=5.3.1
RUN pip install requests>=2.32.2
RUN pip install scipy>=1.4.1
RUN pip install thop>=0.1.1
RUN pip install tqdm>=4.66.3
RUN pip install ultralytics>=8.2.34
RUN pip install pandas>=1.1.4
RUN pip install seaborn>=0.11.0
RUN pip install setuptools>=70.0.0

# Install uvicorn globally
RUN pip install uvicorn

# Expose the port (adjust based on your app's port)
EXPOSE 8000

# Set the default command to run your app (adjust if necessary)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
