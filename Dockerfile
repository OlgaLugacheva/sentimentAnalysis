FROM ubuntu:latest
LABEL authors="grse"

ENTRYPOINT ["top", "-b"]