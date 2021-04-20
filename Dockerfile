FROM continuumio/miniconda3:4.9.2

# Create environment
WORKDIR /home
COPY src src
COPY pretrained_weights pretrained_weights
COPY start.sh .
COPY environment.yaml .
RUN conda env create -f environment.yaml
ENV FLASK_APP=src/server.py

# Expose default flask port
EXPOSE 5000

# Run
CMD ["conda", "run", "--no-capture-output", "-n", "mantranet", "/bin/bash", "-c", "./start.sh"]
