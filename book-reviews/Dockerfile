FROM python:2.7.13-slim

WORKDIR /work

ADD . .

RUN mkdir /data && \
    pip install BeautifulSoup

ENTRYPOINT ["./convert_all.sh"]
CMD [""]
