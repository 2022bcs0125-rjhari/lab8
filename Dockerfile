FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn

CMD ["python","src/train.py"]