FROM  python:3.11-slim-buster
LABEL Name=mlopssd Version=0.0.1
RUN apt-get -y update && apt-get install -y fortunes
CMD ["sh", "-c", "/usr/games/fortune -a | cowsay"]
#RUN dvc init
#COPY .dvc/config .dvc/config
#COPY *.dvc .dvc/
#RUN dvc config core.no_scm true
#COPY <path_to_default.json> default.json
#dvc remote modify myremote --local gdrive_service_account_json_file_path default.json
#RUN dvc pull
