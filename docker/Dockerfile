FROM python:3.9.17
WORKDIR /digits
COPY . /digits
RUN pip3 install -r /digits/requirements.txt
VOLUME /digits/models
CMD ["pytest"]
